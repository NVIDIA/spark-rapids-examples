/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

import org.apache.hadoop.yarn.server.nodemanager.api.deviceplugin.Device;
import org.apache.hadoop.yarn.server.nodemanager.api.deviceplugin.DeviceRuntimeSpec;
import org.apache.hadoop.yarn.server.nodemanager.api.deviceplugin.YarnRuntimeType;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Test case for NvidiaGPUMigPluginForRuntimeV2 device plugin.
 */
public class TestNvidiaGPUMigPluginForRuntimeV2 {

    private static final Logger LOG =
            LoggerFactory.getLogger(TestNvidiaGPUMigPluginForRuntimeV2.class);

    @Test
    public void testGetNvidiaDevices() throws Exception {
        NvidiaGPUMigPluginForRuntimeV2.NvidiaCommandExecutor mockShell =
                mock(NvidiaGPUMigPluginForRuntimeV2.NvidiaCommandExecutor.class);
        String deviceInfoShellOutput =
                "0, 00000000:04:00.0, [N/A]\n" +
                "1, 00000000:82:00.0, Enabled";
        String majorMinorNumber0 = "c3:0";
        String majorMinorNumber1 = "c3:1";
        String deviceMigInfoShellOutput =
                "GPU 0: NVIDIA A100 80GB PCIe (UUID: GPU-aa72194b-fdd4-24b0-f659-17c929f46267)\n" +
                "  MIG 1g.10gb     Device  0: (UUID: MIG-aa2c982c-48a9-5046-b7f8-aa4732879e02)\n" +
                "GPU 1: NVIDIA A100 80GB PCIe (UUID: GPU-aa7153bf-c0ba-00ef-cdce-f861c34172f6)\n" +
                "  MIG 1g.10gb     Device  0: (UUID: MIG-aa59d467-ba39-5d0a-a085-66af03246526)\n" +
                "  MIG 1g.10gb     Device  1: (UUID: MIG-aad5cb29-8e6f-510a-8352-8e18f483dc74)" +
        when(mockShell.getDeviceInfo()).thenReturn(deviceInfoShellOutput);
        when(mockShell.getDeviceMigInfo()).thenReturn(deviceMigInfoShellOutput);
        when(mockShell.getMajorMinorInfo("nvidia0"))
                .thenReturn(majorMinorNumber0);
        when(mockShell.getMajorMinorInfo("nvidia1"))
                .thenReturn(majorMinorNumber1);
        NvidiaGPUMigPluginForRuntimeV2 plugin = new NvidiaGPUMigPluginForRuntimeV2();
        plugin.setShellExecutor(mockShell);
        plugin.setPathOfGpuBinary("/fake/nvidia-smi");

        Set<Device> expectedDevices = new TreeSet<>();
        expectedDevices.add(Device.Builder.newInstance()
                .setId(0).setHealthy(true)
                .setBusID("00000000:04:00.0")
                .setDevPath("/dev/nvidia0")
                .setMajorNumber(195)
                .setStatus("0")
                .setMinorNumber(0).build());
        expectedDevices.add(Device.Builder.newInstance()
                .setId(1).setHealthy(true)
                .setBusID("00000000:82:00.0")
                .setDevPath("/dev/nvidia1")
                .setMajorNumber(195)
                .setStatus("0")
                .setMinorNumber(1).build());
        expectedDevices.add(Device.Builder.newInstance()
                .setId(2).setHealthy(true)
                .setBusID("00000000:82:00.0")
                .setDevPath("/dev/nvidia1")
                .setMajorNumber(195)
                .setStatus("1")
                .setMinorNumber(1).build());
        Set<Device> devices = plugin.getDevices();
        Assert.assertEquals(expectedDevices, devices);
    }

    @Test(expected = Exception.class)
    public void testOnDeviceAllocatedMultiGPU() throws Exception {
        NvidiaGPUMigPluginForRuntimeV2 plugin = new NvidiaGPUMigPluginForRuntimeV2();
        Set<Device> allocatedDevices = new TreeSet<>();

        DeviceRuntimeSpec spec = plugin.onDevicesAllocated(allocatedDevices,
                YarnRuntimeType.RUNTIME_DEFAULT);
        Assert.assertNull(spec);

        // allocate one device
        allocatedDevices.add(Device.Builder.newInstance()
                .setId(0).setHealthy(true)
                .setBusID("00000000:04:00.0")
                .setDevPath("/dev/nvidia0")
                .setMajorNumber(195)
                .setMinorNumber(0).build());
        spec = plugin.onDevicesAllocated(allocatedDevices,
                YarnRuntimeType.RUNTIME_DOCKER);
        Assert.assertEquals("nvidia", spec.getContainerRuntime());
        Assert.assertEquals("0", spec.getEnvs().get("NVIDIA_VISIBLE_DEVICES"));

        // two device allowed
        allocatedDevices.add(Device.Builder.newInstance()
                .setId(0).setHealthy(true)
                .setBusID("00000000:82:00.0")
                .setDevPath("/dev/nvidia1")
                .setMajorNumber(195)
                .setMinorNumber(1).build());
        spec = plugin.onDevicesAllocated(allocatedDevices,
                YarnRuntimeType.RUNTIME_DOCKER);
    }

    @Test
    public void testMultiGPUsEnvPrecedence() throws Exception {
        NvidiaGPUMigPluginForRuntimeV2 plugin = new NvidiaGPUMigPluginForRuntimeV2();
        Set<Device> allocatedDevices = new TreeSet<>();

        DeviceRuntimeSpec spec = plugin.onDevicesAllocated(allocatedDevices,
                YarnRuntimeType.RUNTIME_DEFAULT);
        Assert.assertNull(spec);

        // allocate one device
        allocatedDevices.add(Device.Builder.newInstance()
                .setId(0).setHealthy(true)
                .setBusID("00000000:04:00.0")
                .setDevPath("/dev/nvidia0")
                .setMajorNumber(195)
                .setMinorNumber(0).build());

        // two device allowed
        allocatedDevices.add(Device.Builder.newInstance()
                .setId(0).setHealthy(true)
                .setBusID("00000000:82:00.0")
                .setDevPath("/dev/nvidia1")
                .setMajorNumber(195)
                .setMinorNumber(1).build());

        // test that env variable takes presedence
        plugin.setShouldThrowOnMultipleGPUFromConf(true);
        Map<String, String> envs = new HashMap<>();
        envs.put("NVIDIA_MIG_PLUGIN_THROW_ON_MULTIPLE_GPUS", "false");
        // note the allocated devices doesn't matter here, just the env passed in
        plugin.allocateDevices(allocatedDevices, 2, envs);
        spec = plugin.onDevicesAllocated(allocatedDevices,
                YarnRuntimeType.RUNTIME_DOCKER);
        Assert.assertEquals("nvidia", spec.getContainerRuntime());
        Assert.assertEquals("0,1", spec.getEnvs().get("NVIDIA_VISIBLE_DEVICES"));
    }

    @Test
    public void testMultiGPUsConf() throws Exception {
        NvidiaGPUMigPluginForRuntimeV2 plugin = new NvidiaGPUMigPluginForRuntimeV2();
        Set<Device> allocatedDevices = new TreeSet<>();

        DeviceRuntimeSpec spec = plugin.onDevicesAllocated(allocatedDevices,
                YarnRuntimeType.RUNTIME_DEFAULT);
        Assert.assertNull(spec);

        // allocate one device
        allocatedDevices.add(Device.Builder.newInstance()
                .setId(0).setHealthy(true)
                .setBusID("00000000:04:00.0")
                .setDevPath("/dev/nvidia0")
                .setMajorNumber(195)
                .setMinorNumber(0).build());

        // two device allowed
        allocatedDevices.add(Device.Builder.newInstance()
                .setId(0).setHealthy(true)
                .setBusID("00000000:82:00.0")
                .setDevPath("/dev/nvidia1")
                .setMajorNumber(195)
                .setMinorNumber(1).build());

        // test that env variable takes presedence
        plugin.setShouldThrowOnMultipleGPUFromConf(false);
        spec = plugin.onDevicesAllocated(allocatedDevices,
                YarnRuntimeType.RUNTIME_DOCKER);
        Assert.assertEquals("nvidia", spec.getContainerRuntime());
        Assert.assertEquals("0,1", spec.getEnvs().get("NVIDIA_VISIBLE_DEVICES"));
    }

    @Test
    public void testOnDeviceAllocatedMig() throws Exception {
        NvidiaGPUMigPluginForRuntimeV2 plugin = new NvidiaGPUMigPluginForRuntimeV2();
        Set<Device> allocatedDevices = new TreeSet<>();

        DeviceRuntimeSpec spec = plugin.onDevicesAllocated(allocatedDevices,
                YarnRuntimeType.RUNTIME_DEFAULT);
        Assert.assertNull(spec);

        Map<Integer, String> testMigDevices = new HashMap<>();
        testMigDevices.put(0, "0");
        plugin.setMigDevices(testMigDevices);

        // allocate one device
        allocatedDevices.add(Device.Builder.newInstance()
                .setId(0).setHealthy(true)
                .setBusID("00000000:04:00.0")
                .setDevPath("/dev/nvidia0")
                .setMajorNumber(195)
                .setMinorNumber(0).build());
        spec = plugin.onDevicesAllocated(allocatedDevices,
                YarnRuntimeType.RUNTIME_DOCKER);
        Assert.assertEquals("nvidia", spec.getContainerRuntime());
        Assert.assertEquals("0:0", spec.getEnvs().get("NVIDIA_VISIBLE_DEVICES"));
    }

    @Test
    public void testOnDeviceAllocatedNoMig() throws Exception {
        NvidiaGPUMigPluginForRuntimeV2 plugin = new NvidiaGPUMigPluginForRuntimeV2();
        Set<Device> allocatedDevices = new TreeSet<>();

        DeviceRuntimeSpec spec = plugin.onDevicesAllocated(allocatedDevices,
                YarnRuntimeType.RUNTIME_DEFAULT);
        Assert.assertNull(spec);

        // allocate one device
        allocatedDevices.add(Device.Builder.newInstance()
                .setId(0).setHealthy(true)
                .setBusID("00000000:04:00.0")
                .setDevPath("/dev/nvidia0")
                .setMajorNumber(195)
                .setMinorNumber(0).build());
        spec = plugin.onDevicesAllocated(allocatedDevices,
                YarnRuntimeType.RUNTIME_DOCKER);
        Assert.assertEquals("nvidia", spec.getContainerRuntime());
        Assert.assertEquals("0", spec.getEnvs().get("NVIDIA_VISIBLE_DEVICES"));
    }
}
