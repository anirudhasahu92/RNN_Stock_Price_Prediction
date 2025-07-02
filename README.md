# Stock Price Prediction Using RNNs

## Objective

The objective of this project is to try and predict the stock prices using historical data from four technology companies: IBM, Google (GOOGL), Amazon (AMZN), and Microsoft (MSFT). By using data from multiple companies in the same sector, we aim to capture broader market sentiment and potentially improve prediction accuracy.

The problem statement can be summarized as follows: Given the stock prices of Amazon, Google, IBM, and Microsoft for a set number of days, predict the stock price of these companies after that window.

## Business Value

Stock market data is inherently sequential, making Recurrent Neural Networks (RNNs) well-suited for this type of analysis. Tracking metrics like open, close, high, low prices, and volume provides a rich time series dataset. Identifying patterns within this data is not only academically interesting but also holds significant financial value, as accurate predictions can lead to profitable investment strategies.

## Data Description

The project utilizes four CSV files, one for each stock (AMZN, GOOGL, IBM, MSFT). These files contain historical data from January 1, 2006, to January 1, 2018, sourced from NYSE and NASDAQ. Each file includes the following columns:

- `Date`: The date of the record.
- `Open`: The stock price at market open.
- `High`: The highest stock price during the day.
- `Low`: The lowest stock price during the day.
- `Close`: The stock price at market close.
- `Volume`: The total number of shares traded.
- `Name`: The stock ticker symbol.

Each dataset contains 3019 records.

## 1 Data Loading and Preparation

This section covers loading the individual company stock data, aggregating it into a single DataFrame, handling missing values, and performing initial data analysis and visualization.

### 1.1 Data Aggregation

The data from the four individual CSV files is combined into a single DataFrame. This involves:
- Reading each CSV into a DataFrame.
- Selecting and renaming the `Close` price column with the company name.
- Merging the DataFrames based on the `Date` column.
- Converting the `Date` column to datetime objects and sorting the DataFrame.

Missing values are identified and handled by dropping rows with any `NaN` values.

### 1.2 Analysis and Visualisation

Exploratory data analysis is performed to understand the characteristics of the data.
- **Volume Analysis:** The frequency distribution of stock volumes for each company is visualized using histograms. A line plot shows the variation of trading volume over time for each stock.
- **Correlation Analysis:** Line plots visualize the variation of closing, opening, high, and low prices over time for all stocks. Heatmaps are used to visualize the correlation matrix of closing, opening, high, and low prices among the four stocks. This helps in understanding the relationships between the stock movements.

### 1.3 Data Processing

The data is preprocessed to be suitable for RNN models, which work with sequential data.
- **Windowing:** A `create_windows` function is defined to create input sequences (windows) and corresponding target values from the time series data. A window size of 30 days is chosen after observing potential patterns in monthly slices of the data.
- **Scaling:** A `scale_series_partial` function (though global scaling was eventually used for final evaluation) is initially considered for scaling data within windows to prevent data leakage.
- **Train-Test Split:** A `split_train_val` function splits the windowed data into training and validation sets.

These functions are combined in `prepare_stock_data` to generate the final scaled and windowed training and validation datasets.

## 2 RNN Models

This section focuses on building, tuning, and evaluating different RNN models for stock price prediction.

### 2.1 Simple RNN Model

A basic Simple RNN model is built using a single `SimpleRNN` layer followed by a `Dense` layer.

Hyperparameter tuning is performed using KerasTuner `RandomSearch` to find the optimal number of units and dropout rate for the Simple RNN model for each stock.

The best performing Simple RNN model for each stock is then loaded and evaluated on the validation set. Predictions are inverse-scaled to compare with the actual stock prices.

### 2.2 Advanced RNN Models

This section explores more advanced RNN architectures like LSTM and GRU.

A `build_advanced_rnn` function is created to build models using `LSTM`, `GRU`, or `SimpleRNN` layers, allowing for tuning of units and dropout.

LSTM and GRU models are built and trained for each stock. Predictions are made on the validation set and inverse-scaled for comparison and evaluation.

Hyperparameter tuning using KerasTuner can also be applied to LSTM and GRU models in a similar fashion to the Simple RNN.

The optimal models found through tuning (e.g., `best_RNN_MSFT.h5`, `best_RNN_IBM.h5`) are loaded, and their performance is evaluated on the validation data using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²). Plots showing the comparison of actual and predicted prices are generated.

## 3 Predicting Multiple Target Variables (Optional)

This section is outlined for exploring the possibility of predicting the closing prices of all four companies simultaneously using a multi-output RNN model. Due to time constraints, this part of the project was not completed, but the steps would involve:
- Preparing data with multiple target columns.
- Building RNN models with multiple output neurons.
- Tuning hyperparameters for the multi-output models.
- Evaluating the performance across all predicted stocks.

## 4 Conclusion

### 4.1 Conclusion and insights

This project successfully demonstrates the application of Recurrent Neural Networks (Simple RNN, LSTM, and GRU) for predicting stock prices using historical data from multiple technology companies.

The data aggregation and preprocessing steps, including handling missing values and creating windowed sequences, were crucial for preparing the time series data for RNNs. The exploratory data analysis provided valuable insights into the data's characteristics and the correlations between the stock prices of different companies. The strong positive correlation observed among the stock prices suggests that using data from multiple companies in the same sector is a reasonable approach and could potentially help the models capture sector-wide trends.

Hyperparameter tuning using KerasTuner helped identify better network configurations for the Simple RNN models, leading to improved performance.

While the Simple RNN models showed some ability to capture the general trend of the stock prices, the plots and evaluation metrics (MAE, RMSE, R²) indicate that the predictions are not perfectly aligned with the actual values. Factors like market volatility, unforeseen events, and the inherent complexity of financial markets make accurate stock price prediction a challenging task. The performance metrics provide a quantitative measure of the model's accuracy, with lower MAE and RMSE indicating better predictions and a higher R² indicating a better fit to the data.

The advanced RNN models (LSTM and GRU) were also implemented and showed similar performance characteristics, suggesting that for this specific dataset and problem setup, the added complexity of LSTM or GRU might not yield significantly better results compared to a tuned Simple RNN, although further extensive tuning could explore this further.

The forecasting example illustrates how a trained model can be used to predict future stock prices for a specified horizon. While this provides a glimpse into potential future movements, it's important to remember that these are predictions based on past patterns and do not guarantee future outcomes.

**In summary,** this project provides a foundational approach to stock price prediction using RNNs. The results highlight the potential of these models for time series forecasting in finance, while also emphasizing the difficulty of achieving high accuracy in a volatile market. Future work could explore more complex architectures, incorporating external factors (e.g., news sentiment, economic indicators), and employing more advanced feature engineering and scaling techniques to potentially improve predictive performance.
