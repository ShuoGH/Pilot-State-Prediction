# Reducing Commercial Aviation Fatalities

This is the competition of kaggle in [https://www.kaggle.com/c/reducing-commercial-aviation-fatalities/overview].

Train data recorded the data set of pilots in four statments.


To see the data, see the Kaggle [data](https://www.kaggle.com/c/reducing-commercial-aviation-fatalities/data).

| Category | State Name            | Count   |
|----------|-----------------------|---------|
| A        | Baseline              | 2848809 |
| C        | Channelized Attention | 1652686 |
| B        | Startle/Surprise      | 235329  |
| D        | Diverted Attention    | 130597  |

----------
### Project pipeline:

1. EDA
- Draw the plots to see the features.

2. Data Processing 

    - **Under sample**: data is very unbalance (see the table in the beginning). So we did downsample to balance it.
    - **Sample**: decide during the experiment, since the data set is huge.

3. Models
    - **Random Forest**
    - **LightGBM**
    - **KNN**
    - **Neural Network**

4. Evaluate
    - draw the ROC: it's the most popular method to see the performance 