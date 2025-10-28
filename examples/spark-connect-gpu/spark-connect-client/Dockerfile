# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build stage
FROM maven:3.8-openjdk-17 AS builder
# Set platform to linux/amd64
ENV DOCKER_DEFAULT_PLATFORM=linux/amd64
# Copy the entire project
COPY scala /build
WORKDIR /build
# Build all modules
RUN mvn clean install -DskipTests

# Spark connect client image
FROM apache/spark:4.0.0

USER root
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

RUN mkdir -p /home/spark/demo

COPY notebook /home/spark/demo/notebook
COPY scala /home/spark/demo/scala
COPY python /home/spark/demo/python
COPY --from=builder /build/target/*-jar-with-dependencies.jar /home/spark/demo/scala/

RUN chown -R spark:spark /home/spark
RUN chown -R spark:spark /home/spark/demo
RUN usermod -d /home/spark spark

USER spark
WORKDIR /home/spark/demo

SHELL [ "/bin/bash", "-c" ]
ENV SHELL=/bin/bash