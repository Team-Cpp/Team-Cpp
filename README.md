# Intellidock

<p style='text-align: justify;'>

Predicting the WTI crude oil price of tomorrow using machine learning techniques. This software enables you to use a gradient boosted decision tree and long-short term memory neural network to make a educated decision on whether to sell your oil barrels or wait.

</p>

## Quick start guide:

Python version 3.7.x

<p style='text-align: justify;'>

* Clone the repository and source environment.
```
git clone https://github.com/rrStein/Team-Cpp.git
cd Team-Cpp
source env.sh
```
##### The bash script env.sh will check whether you have the necessarry packages installed in your environment to run all the functions of the software. It will also set the path to your copy of the software as an environment variable.

</p>

## (Graphical) User Guide

<p style='text-align: justify;'>

* To use a graphical version of BDT prediction run... 
```
python GUI/Chris_Intellidock.py 
```
The LSTM approach will soon be integrated to the GUI.

Eleonora?

</p>

## BDT Command Line User Guide

<p style='text-align: justify;'>

* To use command line version of BDT prediction, run the following command and follow printed instructions.
    ```
    python BDT_dev/Intellidock_BDT(rrs_cw).py
    ```

Chris?

</p>

## LSTM Command Line User Guide

<p style='text-align: justify;'>

* To use command line version of LSTM prediction, run the following command and follow printed instructions. 
    ```
    python LSTM_dev/predict_lstm.py
    ```
* Refer to LSTM_dev/requirements.txt for a full list of required modules.
* The LSTM NN uses daily close data and looks back 50 days to make a prediction for the following day.
* The variables used to train the model are:
    * [The WTI Crude Oil price itself](https://finance.yahoo.com/quote/CL=F?p=CL=F)
    * [The Oil Production values for the US from EIA](https://www.eia.gov/dnav/pet/hist_xls/WCRFPUS2w.xls)
    * [The Natural Gas price](https://finance.yahoo.com/quote/NG%3DF/)
    * [The Nasdaq composite price](https://finance.yahoo.com/quote/%5EIXIC?p=%5EIXIC)
    * The number of days the WTI Crude Oil price has been above (+) or below (-) its corresponding [200 day simple moving average line.](https://www.investopedia.com/ask/answers/013015/why-200-simple-moving-average-sma-so-common-traders-and-analysts.asp)
    * [The moving average convergence divergence (MACD) of WTI Crude Oil price (12-26)](https://www.investopedia.com/terms/m/macd.asp)
    * [The 14-day momentum of WTI Crude Oil price](https://www.investopedia.com/articles/technical/081501.asp)
* Data is queried using both [quandl](https://www.quandl.com/) and [yahoofinance](https://github.com/ranaroussi/yfinance) for python.
* Data that is not available in daily interval (oil production) is forward filled with values.
* Weekend data or any missing/nonsense values are filled by using the [scipy interpolate method interp1d with the 'nearest' technique](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d) 
* If using a custom datafile to train the model with, then the columns must be labeled as ["Prices", "OilProduction", "NatGasPrices", "Nasdaq", "daysAbove200dSMA", "MACD_12_26" "Momentum_14"]
* In order to use all functionality the script uses:
    1. A input .csv file with the fields described above in daily intervals.
    2. A .h5 model file.
    3. History .csv file that holds the models' training and testing log. 
    4. A min-max scaler file used to normalise the training data.
* The plots of all variables used compared to the WTI Crude Oil price can be viewed by selecting the relevant option when running the script.
* The correlations between the variables can also be viewed on a heatmap by selecting the relevant option when running the script.
* The NN architecture can be viewed by running the script and selecting the option to display model architecture.
  
![Model](https://github.com/rrStein/Team-Cpp/blob/master/LSTM_dev/outputs/model.png)

* Similarly, the predicted vs real price can be plotted for any period of time after 2010 (or whatever cut you have specified for your model) by selecting the relevant option.
* It is also possible to view the training and validation loss plot which may be of use when tinkering with input parameters and training a new model.
* Training of model is set for 200 epochs with a batch size of 20 and early stopping after 30 rounds of no improvements (min delta = 0.0001), meaning it could take a while...
  
    ### Code description for developers

    * The script imports all the required modules at the start.
    * This is followed by setting all parameters used for reading/writing files, training/loading model, data selecting/splitting, creating I/O directories.
    * Subsequently all the code is split to functions which can be called to do all necessarry actions such as download data, train model, make prediction, etc...
    * Finally, there is a loop asking for user input to run the different functions.

    #### Functions:
    - **oilProduction** - Gets data from EIA about oil production in the US (weekly)
    - **combineFrames** - A legacy function to combine WTI Price data with Oil Production data into one pandas dataframe. Not strictly necessary anymore but it works...
    - **CALCULATIONS (sma, ema, bollinger, momentum, relative_strength_index, rate_of_change, macd)** - Used to calculate the respective techincal indicators for a stock price.
    - **profitNextDayObjective** - Legacy function to redefine the objective function to predict as whether it was profitable to hold or not.
    - **create_features** - A function to trim the dataframe for specified features or introduce lag in specified variables.
    - **getData** - Gets data from quandl and/or yfinance if given a key of a variable from the dictionary defined in the beginning of script.
    - **build_timeseries** - Transforms data(frame) into suitable format for training the LSTM based on how many timesteps have been specified. See comments for more info.
    - **plot2axis** - A base/general plot function to make a base plot with 2 axis and the WTI price overlayed with its bollinger bands and 20 day SMA.
    - **plot_data** - Plots the specified variables for comparison with the WTI Crude Oil price using the plot2axis function and saves the figures.
    - **plot_correlations** - Plots a heat map of the correlations between the specified features.
    - **trim_dataset** - Trims the LSTM input dataset so that it would be divisible by the batch size defined for the training.
    - **create_model** - Creates the base structure of the LSTM model to be trained. Layers, neurons, dropouts, kernels, optimisers are all set here. This can be used to later make a new model with same weights that uses a different batch size which is necessary if one wishes to predict a different number of steps than the training batch size.
    - **train_model** - This transforms input dataframe into suitable form for training an LSTM, scales the training data with a [Min-Max scaler](https://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/) and saves the scaler, trains the created model with the transformed data, saves training history, and also outputs the weights of the resulting model.
    - **test_train_loss** - Takes the model training history dataframe as input and plots the test-train loss graph.
    - **visualise_prediction** - Takes data and the desired interval (specified by how many days ago to start and end predicting) to compare visually the predicted and real values for the WTI Crude Oil price.
    - **predict_new** - Takes the trained model weights, data, and number of days wished to predict ahead (currently limited to 1 i.e. tomorrow) and resets the model by making a new identical model with same weights but batch size == days and produces the sought predictions of WTI Crude Oil price.
    - **getLstmData** - Calls the relevant functions to query data, calculate variables, and combine all of it to produce the input dataframe and saves it to file.
    - **plot_model** - This is a keras.utils module that plots the models' architecture.

  


</p>

## To-Do list

<p style='text-align: justify;'>

We have developed all the core functionality and basic integration with a GUI but some tasks remain to make this a easy to use and effective software package. The order of these tasks may not represent the actual importance... (maybe these could be raised as issues on github?)

1. Integrate all functionality with the GUI
2. Develop the GUI to be more user-friendly and pretty
3. Find a sensible way to combine the LSTM and BDT output
4. Make a container for the whole software with minimal dependencies
5. Move all backend (training, predicting, data collection) to the cloud so that it could be ran by calling actions on the frontend (GUI) on a local machine with minimal bandwidth requirement.
6. Complete User Guide and Documentation.
7. Confidence bands / probabilities for predictions
8. Test the profitability of using LSTM for making sell/hold decisions
9. Introduce lag to the LSTM to make longer term predictions (maybe up to a week ahead?)

</p>
