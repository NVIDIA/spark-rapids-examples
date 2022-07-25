# How to download the Mortgage dataset



## Steps to download the data

1. Go to the [Fannie Mae](https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data) website
2. Click on [Single-Family Loan Performance Data](https://datadynamics.fanniemae.com/data-dynamics/?&_ga=2.181456292.2043790680.1657122341-289272350.1655822609#/reportMenu;category=HP)
    * Register as a new user if you are using the website for the first time
    * Use the credentials to login
3. Select [HP](https://datadynamics.fanniemae.com/data-dynamics/#/reportMenu;category=HP)
4. Click on  **Download Data** and choose *Single-Family Loan Performance Data*
5. You will find a tabular list of 'Acquisition and Performance' files sorted based on year and quarter. Click on the file to download `Eg: 2017Q1.zip`
6. Unzip the downlad file to extract the csv file `Eg: 2017Q1.csv`
7. Copy only the csv files to a new folder for the ETL to read

## Notes
1. Refer to the [Loan Performance Data Tutorial](https://capitalmarkets.fanniemae.com/media/9066/display) for more details. 
2. Note that *Single-Family Loan Performance Data* has 2 componenets. However, the Mortgage ETL requires only the first one (primary dataset)
    * Primary Dataset:  Acquisition and Performance Files
    * HARP Dataset
3. Use the [Resources](https://datadynamics.fanniemae.com/data-dynamics/#/resources/HP) section to know more about the dataset