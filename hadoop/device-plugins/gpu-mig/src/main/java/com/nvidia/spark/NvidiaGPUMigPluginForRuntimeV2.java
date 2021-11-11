/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.Shell;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.hadoop.yarn.server.nodemanager.api.deviceplugin.Device;
import org.apache.hadoop.yarn.server.nodemanager.api.deviceplugin.DevicePlugin;
import org.apache.hadoop.yarn.server.nodemanager.api.deviceplugin.DevicePluginScheduler;
import org.apache.hadoop.yarn.server.nodemanager.api.deviceplugin.DeviceRegisterRequest;
import org.apache.hadoop.yarn.server.nodemanager.api.deviceplugin.DeviceRuntimeSpec;
import org.apache.hadoop.yarn.server.nodemanager.api.deviceplugin.YarnRuntimeType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Nvidia GPU plugin supporting both Nvidia container runtime v2.
 * It supports discovering and allocating MIG devices. Currently, with CUDA 11,
 * only enumeration of a single MIG instance is supported. This means that
 * this plugin officially only supports 1 GPU per container and by default
 * will throw an exception if more are requested. The behavior of throwing
 * an exception is configurable by either setting the environment variable
 * {@code NVIDIA_MIG_PLUGIN_THROW_ON_MULTIPLE_GPUS} or by setting the YARN config
 * {@code com.nvidia.spark.NvidiaGPUMigPluginForRuntimeV2.throwOnMultipleGPUs}
 * to false.
 */
public class NvidiaGPUMigPluginForRuntimeV2 implements DevicePlugin,
        DevicePluginScheduler {
    public static final Logger LOG = LoggerFactory.getLogger(
            NvidiaGPUMigPluginForRuntimeV2.class);

    public static final String NV_RESOURCE_NAME = "nvidia/miggpu";

    private NvidiaCommandExecutor shellExecutor = new NvidiaCommandExecutor();

    private Map<String, String> environment = new HashMap<>();

    // If this environment is set, use it directly
    private static final String ENV_BINARY_PATH = "NVIDIA_SMI_PATH";

    private static final String DEFAULT_BINARY_NAME = "nvidia-smi";

    private static final String DEV_NAME_PREFIX = "nvidia";

    private static final String THROW_MULTI_CONF =
            "com.nvidia.spark.NvidiaGPUMigPluginForRuntimeV2.throwOnMultipleGPUs";

    private static final String THROW_MULTI_ENV = "NVIDIA_MIG_PLUGIN_THROW_ON_MULTIPLE_GPUS";

    private Boolean shouldThrowOnMultipleGPUFromConf =
        new Configuration().getBoolean(THROW_MULTI_CONF, true);
    private String shouldThrowOnMultipleGPUFromEnv = null;

    private String pathOfGpuBinary = null;

    // command should not run more than 10 sec.
    private static final int MAX_EXEC_TIMEOUT_MS = 10 * 1000;

    // When executable path not set, try to search default dirs
    // By default search /usr/bin, /bin, and /usr/local/nvidia/bin (when
    // launched by nvidia-docker.
    private static final String[] DEFAULT_BINARY_SEARCH_DIRS = new String[]{
            "/usr/bin", "/bin", "/usr/local/nvidia/bin"};

    // device id -> mig id, populated during discovery and used when launching
    // containers
    private Map<Integer, String> migDevices = new HashMap<>();

    private String migInfoOutput = null;


    @Override
    public DeviceRegisterRequest getRegisterRequestInfo() throws Exception {
        return DeviceRegisterRequest.Builder.newInstance()
                .setResourceName(NV_RESOURCE_NAME).build();
    }

    @Override
    public Set<Device> getDevices() throws Exception {
        shellExecutor.searchBinary();
        TreeSet<Device> r = new TreeSet<>();
        String output;
        try {
            output = shellExecutor.getDeviceInfo();
            String[] lines = output.trim().split("\n");
            int id = 0;
            for (String oneLine : lines) {
                String[] tokensEachLine = oneLine.split(",");
                if (tokensEachLine.length != 3) {
                    throw new Exception("Cannot parse the output to get the MIG enabled info. "
                            + "output: " + oneLine + " expected index,pci.bus_id,mig.mode.current");
                }
                String minorNumber = tokensEachLine[0].trim();
                String busId = tokensEachLine[1].trim();
                String migMode = tokensEachLine[2].trim();
                String majorNumber = getMajorNumber(DEV_NAME_PREFIX
                        + minorNumber);

                if (majorNumber != null) {
                    if (migMode.equalsIgnoreCase("enabled")) {
                        if (migInfoOutput == null) {
                            // we get the mig info for all the GPUs on the host so only get it once
                            migInfoOutput = shellExecutor.getDeviceMigInfo();
                            if (migInfoOutput == null) {
                                throw new Exception("MIG device enabled but no device info found");
                            }
                        }
                        String[] linesMig = migInfoOutput.trim().split("\n");
                        Integer minorNumInt = Integer.parseInt(minorNumber);
                        Integer migDevCount = 0;
                        Integer numMigOutputLines = linesMig.length;
                        for (int idmig = 0; idmig < numMigOutputLines; idmig++) {
                            // first line should start with GPU
                            // GPU 0: NVIDIA A30 (UUID: GPU-e7076666-0544-e103-4f65-a047fc18269e)
                            // MIG 1g.6gb      Device  0: (UUID: MIG-de9876e2-eef7-5b5a-9701-db694ffe8a77)
                            if (linesMig[idmig].startsWith("GPU " + minorNumInt) && numMigOutputLines > (idmig + 1)) {
                                // process any MIG devices, this expects all the lines to be MIG devices until
                                // we find one that starts with GPU
                                String nextLine = linesMig[++idmig].trim();
                                String regex = "MIG (.+)Device\\s+(\\d+):\\s+\\(UUID:(.*)\\)";
                                Pattern pattern = Pattern.compile(regex);
                                while (nextLine.startsWith("MIG")) {
                                    Matcher matcher = pattern.matcher(nextLine);
                                    while (matcher.find()) {
                                        String devId = matcher.group(2);
                                        migDevices.put(id, devId);
                                        migDevCount++;
                                        r.add(Device.Builder.newInstance()
                                                .setId(id)
                                                .setMajorNumber(Integer.parseInt(majorNumber))
                                                .setMinorNumber(minorNumInt)
                                                .setBusID(busId)
                                                .setDevPath("/dev/" + DEV_NAME_PREFIX + minorNumber)
                                                .setHealthy(true)
                                                .setStatus(devId)
                                                .build());
                                        id++;
                                        if (++idmig < numMigOutputLines) {
                                            nextLine = linesMig[idmig].trim();
                                        } else {
                                            nextLine = "";
                                        }
                                    }
                                }
                                idmig = numMigOutputLines;
                            }
                        }
                        if (migDevCount < 1) {
                            throw new IOException("Error finding MIG devices on GPU with " +
                                "MIG enabled: " + migInfoOutput);
                        }
                        LOG.info("Added GPU " + majorNumber + ":" + minorNumInt +
                            " with MIG Enabled, found " + migDevCount + " MIG devices");
                    } else {
                        Integer majorNumInt = Integer.parseInt(majorNumber);
                        Integer minorNumInt = Integer.parseInt(minorNumber);
                        r.add(Device.Builder.newInstance()
                                .setId(id)
                                .setMajorNumber(majorNumInt)
                                .setMinorNumber(minorNumInt)
                                .setBusID(busId)
                                .setDevPath("/dev/" + DEV_NAME_PREFIX + minorNumber)
                                .setHealthy(true)
                                .build());
                        LOG.info("Added GPU " + majorNumInt + ":" + minorNumInt);
                        id++;
                    }
                }
            }
            return r;
        } catch (IOException e) {
            LOG.debug("Failed to get output from {}", pathOfGpuBinary);
            throw new YarnException(e);
        }
    }

    private Boolean shouldThrowOnMultipleGPUs() {
        // env setting takes highest priority if it is set
        if (shouldThrowOnMultipleGPUFromEnv != null) {
            return Boolean.parseBoolean(shouldThrowOnMultipleGPUFromEnv);
        }
        return shouldThrowOnMultipleGPUFromConf;
    }

    @Override
    public DeviceRuntimeSpec onDevicesAllocated(Set<Device> allocatedDevices,
                                                YarnRuntimeType yarnRuntime) throws Exception {
        LOG.debug("Generating runtime spec for allocated devices: {}, {}",
                allocatedDevices, yarnRuntime.getName());
        if (allocatedDevices.size() > 1 && shouldThrowOnMultipleGPUs()) {
            throw new YarnException("Allocating more than 1 GPU per container is" +
                    " not supported with use of MIG!");
        }
        if (yarnRuntime == YarnRuntimeType.RUNTIME_DOCKER) {
            String nvidiaRuntime = "nvidia";
            String nvidiaVisibleDevices = "NVIDIA_VISIBLE_DEVICES";
            StringBuffer gpuMinorNumbersSB = new StringBuffer();
            for (Device device : allocatedDevices) {
                Integer minorNum = device.getMinorNumber();
                Integer id = device.getId();
                if (migDevices.containsKey(id)) {
                    gpuMinorNumbersSB.append(minorNum + ":" + migDevices.get(id) + ",");
                } else {
                    gpuMinorNumbersSB.append(minorNum + ",");
                }
            }
            String minorNumbers = gpuMinorNumbersSB.toString();
            LOG.info("Nvidia Docker v2 assigned GPU: " + minorNumbers);
            String deviceStr = minorNumbers.substring(0, minorNumbers.length() - 1);
            return DeviceRuntimeSpec.Builder.newInstance()
                    .addEnv(nvidiaVisibleDevices, deviceStr)
                    .setContainerRuntime(nvidiaRuntime)
                    .build();
        }
        return null;
    }

    @Override
    public void onDevicesReleased(Set<Device> releasedDevices) throws Exception {
        // do nothing
    }

    // Get major number from device name.
    private String getMajorNumber(String devName) {
        String output = null;
        // output "major:minor" in hex
        try {
            LOG.debug("Get major numbers from /dev/{}", devName);
            output = shellExecutor.getMajorMinorInfo(devName);
            String[] strs = output.trim().split(":");
            output = Integer.toString(Integer.parseInt(strs[0], 16));
        } catch (IOException e) {
            String msg =
                    "Failed to get major number from reading /dev/" + devName;
            LOG.warn(msg);
        } catch (NumberFormatException e) {
            LOG.error("Failed to parse device major number from stat output");
            output = null;
        }
        return output;
    }

    @Override
    public Set<Device> allocateDevices(Set<Device> availableDevices, int count,
                                       Map<String, String> envs) {
        Set<Device> allocation = new TreeSet<>();
        String envShouldThrow = envs.get(THROW_MULTI_ENV);
        if (envShouldThrow != null) {
            shouldThrowOnMultipleGPUFromEnv = envShouldThrow;
        }
        // Only officially support 1 GPU per container so don't worry about topology
        // scheduling.
        basicSchedule(allocation, count, availableDevices);
        return allocation;
    }

    public void basicSchedule(Set<Device> allocation, int count,
                              Set<Device> availableDevices) {
        // Basic scheduling
        // allocate all available
        if (count == availableDevices.size()) {
            allocation.addAll(availableDevices);
            return;
        }
        int number = 0;
        for (Device d : availableDevices) {
            allocation.add(d);
            number++;
            if (number == count) {
                break;
            }
        }
    }

    /**
     * A shell wrapper class easy for test.
     */
    public class NvidiaCommandExecutor {

        public String getDeviceInfo() throws IOException {
            return Shell.execCommand(environment,
                    new String[]{pathOfGpuBinary, "--query-gpu=index,pci.bus_id,mig.mode.current",
                            "--format=csv,noheader"}, MAX_EXEC_TIMEOUT_MS);
        }

        public String getDeviceMigInfo() throws IOException {
            return Shell.execCommand(environment,
                    new String[]{pathOfGpuBinary, "-L"}, MAX_EXEC_TIMEOUT_MS);
        }

        public String getMajorMinorInfo(String devName) throws IOException {
            // output "major:minor" in hex
            Shell.ShellCommandExecutor shexec = new Shell.ShellCommandExecutor(
                    new String[]{"stat", "-c", "%t:%T", "/dev/" + devName});
            shexec.execute();
            return shexec.getOutput();
        }

        public void searchBinary() throws Exception {
            if (pathOfGpuBinary != null) {
                LOG.info("Skip searching, the NVIDIA gpu binary is already set: "
                        + pathOfGpuBinary);
                return;
            }
            // search env for the binary
            String envBinaryPath = System.getenv(ENV_BINARY_PATH);
            if (null != envBinaryPath) {
                if (new File(envBinaryPath).exists()) {
                    pathOfGpuBinary = envBinaryPath;
                    LOG.info("Use NVIDIA gpu binary: " + pathOfGpuBinary);
                    return;
                }
            }
            LOG.debug("Search binary..");
            // search if binary exists in default folders
            File binaryFile;
            boolean found = false;
            for (String dir : DEFAULT_BINARY_SEARCH_DIRS) {
                binaryFile = new File(dir, DEFAULT_BINARY_NAME);
                if (binaryFile.exists()) {
                    found = true;
                    pathOfGpuBinary = binaryFile.getAbsolutePath();
                    LOG.info("Found binary:" + pathOfGpuBinary);
                    break;
                }
            }
            if (!found) {
                LOG.error("No binary found from env variable: "
                        + ENV_BINARY_PATH + " or path "
                        + DEFAULT_BINARY_SEARCH_DIRS.toString());
                throw new Exception("No binary found for "
                        + NvidiaGPUMigPluginForRuntimeV2.class);
            }
        }
    }

    // visible for testing
    public void setPathOfGpuBinary(String pOfGpuBinary) {
        this.pathOfGpuBinary = pOfGpuBinary;
    }

    // visible for testing
    public void setShellExecutor(NvidiaCommandExecutor shellExecutor) {
        this.shellExecutor = shellExecutor;
    }

    // visible for testing
    public void setMigDevices(Map<Integer, String> migDevices) {
        this.migDevices = migDevices;
    }

    // visible for testing
    public void setShouldThrowOnMultipleGPUFromConf(Boolean shouldThrow) {
        this.shouldThrowOnMultipleGPUFromConf = shouldThrow;
    }
}
