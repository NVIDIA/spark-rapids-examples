# How to run optuna on Spark

## Setup DataBase for Optuna

Optuna offers an RDBStorage option which allows for the persistence of experiments
across different machines and processes, thereby enabling Optuna tasks to be distributed.

This guide will walk you through setting up MySQL as the backend for RDBStorage in Optuna.

We highly recommend installing MySQL on the driver node. This setup eliminates concerns
regarding MySQL connectivity between worker nodes and the driver, simplifying the
management of database connections.

1. Install MySql

``` shell
sudo apt install mysql-server
```

2. Configure MySql

in `/etc/mysql/mysql.conf.d/mysqld.cnf`

``` shell
bind-address    = THE_DRIVER_HOST_IP
mysqlx-bind-address = THE_DRIVER_HOST_IP
```

3. Restart MySql

``` shell
sudo systemctl restart mysql.service
```

4. Setup user

```shell
sudo mysql
```

``` mysql
mysql> CREATE USER 'optuna_user'@'%' IDENTIFIED BY 'optuna_password';
Query OK, 0 rows affected (0.01 sec)

mysql> GRANT ALL PRIVILEGES ON *.* TO 'optuna_user'@'%' WITH GRANT OPTION;
Query OK, 0 rows affected (0.01 sec)

mysql> FLUSH PRIVILEGES;
Query OK, 0 rows affected (0.01 sec)

mysql> EXIT;
Bye
```

Troubleshooting
> If you encounter
`"ERROR 2002 (HY000): Can't connect to local MySQL server through socket '/tmp/mysql.sock' (2)"`, try below commands
> Try `ln -s /var/run/mysqld/mysqld.sock /tmp/mysql.sock`

## Setup optuna python environment

``` shell
sudo apt install libmysqlclient-dev

conda create -n optuna-spark python==3.10
conda activiate optuna-spark
pip install mysqlclient
pip install optuna joblib

# We must install joblibspark from source due to https://github.com/joblib/joblib-spark/issues/51
git clone git@github.com:joblib/joblib-spark.git
cd joblib-spark; pip install .
```

## Create optuna database and study.

On the driver node, run the following commands to establish a database in MySql and create
an Optuna study.

``` shell
mysql -u optuna_user -p -e "CREATE DATABASE IF NOT EXISTS optuna"
optuna create-study --study-name "optuna-spark" --storage "mysql://optuna_user:optuna_password@localhost/optuna"
```

## Pack the optuna runtime environment and run.

``` shell
conda activiate optuna-spark
pip install conda-pack
conda pack -f -o optuna-env.tar.gz
```

After packing the optuna runtime environment, you can play around the optuna on Spark.

```shell
run-optuna-spark.sh
```

If you would like to distribute optuna xgboost tasks to spark cluster, you can run the following
commands,

``` shell
optuna create-study --study-name "optuna-spark-xgboost" --storage "mysql://optuna_user:optuna_password@localhost/optuna"
run-optuna-spark-xgboost.sh
```

## How does it work

![](images/optuna.svg)

The optuna tasks will be serialized into bytes and distributed to spark workers
to run. So it's the optuna task on the executor side that loads the optuna study from RDBStorage, and then
runs the tuning.

During tuning, the optuna tasks needs to send intermediate result back to RDBStorage to persist,
and ask for the parameters from RDBStorage sampled by optuna on the driver to run.

Each optuna task is a Spark application that has only 1 job, 1 stage, 1 task, and the Spark application
will be submitted on the local threads. So here we can use `n_jobs` when configuring the spark backend
to limit at most how many spark applications can be submitted at the same time.

From the above, we can get that the optuna + spark is the Spark application level parallelism, it's not
task-level parallelism. So for the  xgboost case, we first need to make sure the single xgboost task can 
run on a single node without any CPU/GPU OOM.