#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -x

sudo rm -r /var/lib/apt/lists/*
sudo apt clean && sudo apt update --fix-missing -y

if [[ $DB_IS_DRIVER = "TRUE" ]]; then
    # setup database for optuna on driver

    # install mysql server
    sudo apt install -y mysql-server

    if [[ ! -f "/etc/mysql/mysql.conf.d/mysqld.cnf" ]]; then
        sudo apt remove --purge mysql\*
        sudo apt clean && sudo apt update --fix-missing -y
        sudo apt install -y mysql-server
    fi

    if [[ ! -f "/etc/mysql/mysql.conf.d/mysqld.cnf" ]]; then
        echo "ERROR: MYSQL installation failed"
        exit 1
    fi

    # configure mysql
    BIND_ADDRESS=$DB_DRIVER_IP
    MYSQL_CONFIG_FILE="/etc/mysql/mysql.conf.d/mysqld.cnf"
    sudo sed -i "s/^bind-address\s*=.*/bind-address = $BIND_ADDRESS/" "$MYSQL_CONFIG_FILE"
    sudo sed -i "s/^mysqlx-bind-address\s*=.*/mysqlx-bind-address = $BIND_ADDRESS/" "$MYSQL_CONFIG_FILE"
    sudo systemctl restart mysql.service

    # setup user
    OPTUNA_USER="optuna_user"
    OPTUNA_PASSWORD="optuna_password"
    sudo mysql -u root -e "
        CREATE USER IF NOT EXISTS '$OPTUNA_USER'@'%' IDENTIFIED BY '$OPTUNA_PASSWORD';
        GRANT ALL PRIVILEGES ON *.* TO '$OPTUNA_USER'@'%' WITH GRANT OPTION;
        FLUSH PRIVILEGES;"  
fi


# rapids import
SPARK_RAPIDS_VERSION=25.04.0
curl -L https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/${SPARK_RAPIDS_VERSION}/rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}.jar -o \
    /databricks/jars/rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}.jar

# setup cuda: install cudatoolkit 11.8 via runfile approach
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit
# reset symlink and update library loading paths
rm /usr/local/cuda
ln -s /usr/local/cuda-11.8 /usr/local/cuda

sudo /databricks/python3/bin/pip3 install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu11==25.02.*" "cuml-cu11==25.02.*"

# setup python environment
sudo apt clean && sudo apt update --fix-missing -y
sudo apt install pkg-config
sudo apt install -y libmysqlclient-dev
sudo /databricks/python3/bin/pip3 install --upgrade pip
sudo /databricks/python3/bin/pip3 install mysqlclient xgboost
sudo /databricks/python3/bin/pip3 install optuna joblib joblibspark

if [[ $DB_IS_DRIVER = "TRUE" ]]; then
    # create optuna database and study
    sudo mysql -u $OPTUNA_USER -p$OPTUNA_PASSWORD -e "CREATE DATABASE IF NOT EXISTS optuna;"
fi
set +x
