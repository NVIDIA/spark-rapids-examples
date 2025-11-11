#! /bin/bash

# work for jdk 17
java \
--add-exports=java.base/sun.nio.ch=ALL-UNNAMED \
--add-opens=java.base/java.nio=ALL-UNNAMED \
--add-opens=java.base/java.lang.invoke=ALL-UNNAMED \
--add-opens=java.base/java.util=ALL-UNNAMED \
--add-opens=java.base/sun.security.action=ALL-UNNAMED  \
  -cp spark-connect-demo-1.0-SNAPSHOT-jar-with-dependencies.jar connect

