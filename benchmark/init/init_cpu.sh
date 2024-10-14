#!/bin/bash

set -x # for debugging
if [[ $DB_IS_DRIVER = "TRUE" ]]; then
    # setup database for optuna on driver

    # install mysql server
    sudo apt-get update 
    sudo apt-get install -y mysql-server

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

# setup python environment
sudo apt-get install -y libmysqlclient-dev
sudo /databricks/python3/bin/pip3 install --upgrade pip
sudo /databricks/python3/bin/pip3 install mysqlclient optuna joblib joblibspark pandas xgboost scikit-learn

if [[ $DB_IS_DRIVER = "TRUE" ]]; then
    # create optuna database
    sudo mysql -u $OPTUNA_USER -p$OPTUNA_PASSWORD -e "DROP DATABASE IF EXISTS optuna; CREATE DATABASE optuna;"
    # create optuna study
    /databricks/python3/bin/optuna create-study --study-name "optuna-spark-xgboost-cpu" --storage "mysql://$OPTUNA_USER:$OPTUNA_PASSWORD@$DB_DRIVER_IP/optuna"
fi
set +x
