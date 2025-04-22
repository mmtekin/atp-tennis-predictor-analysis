# ATP Tennis Match Prediction Analysis

By Malik Tekin (mtekin@umich.edu)

An analysis of ATP tennis matches to predict match outcomes and explore patterns in professional tennis. This project applies practical data science techniques to understand and predict tennis match results using historical ATP tour data.

## Introduction

This project analyzes historical ATP tennis match data (after transformation, 6152 rows since every row from original dataset had to be split into 2 rows) to uncover patterns in professional tennis and build a pre‑match predictor of match outcomes. I explore how variables like player ranking, age, and Elo rating affect winning probability, then train a model that—using only these pre‑match stats—forecasts which player will win. Such insights can aid coaches, bettors, and tennis analysts in understanding match dynamics before a ball is struck.

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning Process

The data cleaning process involved several sophisticated transformations to prepare the ATP match data for analysis:

1. **Elo Rating Computation**
   - Implemented a dynamic Elo rating system for all players
   - Processed matches chronologically to maintain accurate rating progression
   - Initial rating of 1500 for new players
   - Ratings update after each match using standard Elo formula with k=32

2. **Data Restructuring**
   - Transformed the original match-centric format to a player-centric format
   - Each match was split into two rows to represent both perspectives (winner and loser)
   - This doubled our dataset size but provides balanced training data for our prediction model

3. **Feature Engineering**
   - Converted tournament dates to datetime format
   - Standardized player attributes (rank, age, hand, rank points)
   - Created symmetric features for both players (p1 and p2)
   - Added binary outcome variable 'did_p1_win'

4. **Final Dataset Structure**
Key features in transformed dataset:
   - Tournament info: date, surface, draw_size, level, round
   - Player 1 stats: rank, age, hand, rank_points, elo
   - Player 2 stats: rank, age, hand, rank_points, elo
   - Outcome: did_p1_win (binary)

This transformation approach ensures our model can learn from both winning and losing perspectives, while maintaining the temporal nature of the Elo rating system.

### Univariate Analysis

Our univariate analysis revealed several interesting distributions in our tennis match dataset:

1. **Player Rankings Distribution**
<iframe
src="assets/rankingsDistr.html"
width="800"
height="600"
frameborder="0"
></iframe>

The distribution of player rankings shows a significant skew towards higher-ranked players (ranks 1-49), with 2,778 matches compared to 1,831 matches for players ranked 50-99. This makes sense as our dataset focuses on major tournaments rather than challenger events.

2. **Age Distribution**
<iframe
src="assets/ageDistr.html"
width="800"
height="600"
frameborder="0"
></iframe>

The age distribution reveals that most players are between 25-29 years old, with a steep drop-off after age 29. This suggests that players' physical prime typically lasts until around age 29.

3. **Court Surface Distribution**
<iframe
src="assets/courtDistr.html"
width="800"
height="600"
frameborder="0"
></iframe>

Hard courts dominate the dataset, followed by clay, with grass courts having the fewest matches. This distribution will be important when evaluating our model's performance across different surfaces.

4. **Player Handedness**
<iframe
src="assets/handDistr.html"
width="800"
height="600"
frameborder="0"
></iframe>

The vast majority of players are right-handed, which aligns with general population statistics. This distribution is particularly interesting given the traditional belief about left-handed players having an advantage.

### Bivariate Analysis

Our bivariate analysis focused on understanding how different factors relate to match outcomes:

1. **Age Difference and Win Rate**
<iframe
src="assets/winRateAgeDiffBin.html"
width="800"
height="600"
frameborder="0"
></iframe>

This analysis reveals that:
- Players tend to have higher win rates when they are 2-8 years younger than their opponents
- A sharp increase in win rate occurs with >10 years age difference
- Being older (especially by >4 years) correlates with lower win rates
- The most significant drop in win rate occurs when a player is >12 years older

2. **Handedness and Win Rate**
<iframe
src="assets/winRatePlayerHandBar.html"
width="800"
height="600"
frameborder="0"
></iframe>

Surprisingly, our analysis shows no significant advantage for left-handed players, contrary to common tennis wisdom. This suggests that handedness might not be a crucial feature for match prediction.

3. **Ranking Difference and Win Rate**
<iframe
src="assets/winRateByRankDiff.html"
width="800"
height="600"
frameborder="0"
></iframe>

The relationship between ranking difference and win probability is strong and clear:
- Players with better rankings (negative rank difference) win approximately 75% of matches
- Equal rankings (~0 difference) correspond to ~50% win rates
- Lower-ranked players (positive rank difference) win less than 25% of matches against opponents ranked 90+ spots higher

This strong correlation makes ranking difference a potentially powerful predictor for our model.

### Interesting Aggregates

I analyzed how player ranking tiers interact with court surfaces to influence win rates:

<iframe
src="assets/winRateBySurfaceAndRankTier.html"
width="800"
height="400"
frameborder="0"
></iframe>

This pivot table reveals several fascinating patterns about the relationship between player ranking and surface-specific performance:

1. **Top 10 Players' Dominance**:
   - Elite players (Top 10) maintain impressive win rates >70% across all surfaces
   - Highest win rates on grass (77%) and clay (76%)
   - Relatively lower win rate on hard courts (71%), suggesting surface might be a slight equalizer

2. **Mid-Tier Performance (11-50)**:
   - Players ranked 11-25 show consistent performance (~60% win rate) across surfaces
   - Rankings 26-50 maintain win rates around 48-51%, with grass courts showing slightly better outcomes

3. **Lower Rankings Impact**:
   - Players ranked 100+ struggle significantly with ~40% win rates across all surfaces
   - The skill gap is most pronounced on clay courts, where ranking appears to be particularly important

This analysis suggests that while top players maintain their advantage across all surfaces, the impact of ranking on win probability varies by surface type. Hard courts, being the most common surface, show slightly more competitive matches across ranking tiers, possibly due to players' greater familiarity with the surface.

### Handling Missing Values

Our data cleaning process was designed to minimize the impact of missing values:

1. **Elo Ratings**
   - New players automatically assigned 1500
   - Ratings evolve naturally through match play
   - No imputation needed as system is self-contained

2. **Player Attributes**
   - Rank and rank points: Used as-is (missing values indicate unranked players)
   - Age and hand: Complete in our dataset
   - Surface: No missing values in selected tournaments

3. **Tournament Information**
   - All selected fields (date, surface, level, round) are complete
   - Draw size is consistently recorded

The transformation process ensured that our final dataset contains no missing values that would impact our prediction task.

## Framing a Prediction Problem

I have formulated our analysis as a **binary classification problem** to predict tennis match outcomes. This framing aligns naturally with the binary nature of tennis matches (win/loss) and builds upon the insights gained from our exploratory analysis.

### Prediction Task
- **Target Variable**: `did_p1_win` (1 if Player 1 wins, 0 if Player 1 loses)
- **Type**: Binary Classification
- **Goal**: Predict match winner before the match begins

### Feature Selection
I carefully selected features that would be known before a match starts, ensuring my model could make real-world predictions:

1. **Player Rankings**
   - Player 1 and 2 ATP rankings (`p1_rank`, `p2_rank`)
   - Ranking points (`p1_rank_points`, `p2_rank_points`)
   - Our earlier analysis showed ranking difference is highly predictive of match outcomes

2. **Player Characteristics**
   - Ages (`p1_age`, `p2_age`)
   - Dominant hands (`p1_hand`, `p2_hand`)
   - Age analysis showed younger players often have an advantage

3. **Match Context**
   - Court surface (`surface`)
   - Our aggregate analysis showed surface affects win rates across ranking tiers

### Evaluation Metric
I chose **accuracy** as my primary evaluation metric for several reasons:
1. My dataset is perfectly balanced due to my transformation approach (each match appears twice with players swapped)
2. False positives and false negatives are equally important in match prediction
3. Accuracy provides a straightforward interpretation: the proportion of matches I predict correctly

This framing allows me to build upon my exploratory insights, particularly the strong relationship between ranking differences and match outcomes, while maintaining practical applicability for real-world match prediction.

## Baseline Model

For my initial prediction approach, I implemented a simple decision tree classifier (with max_depth=3) as a baseline model. This choice provides an interpretable starting point while incorporating both ranking and player characteristics.

### Model Architecture
- **Model Type**: Decision Tree (depth-3)
- **Implementation**: Single sklearn Pipeline combining preprocessing and classification
- **Training/Test Split**: 80/20 split with stratification on the target variable

### Features
The baseline model uses four features:
1. **Quantitative Features (2)**
   - Player 1 ATP Ranking (`p1_rank`)
   - Player 2 ATP Ranking (`p2_rank`)

2. **Nominal Features (2)**
   - Player 1 Handedness (`p1_hand`)
   - Player 2 Handedness (`p2_hand`)

### Preprocessing Pipeline
All preprocessing steps are encapsulated in a single sklearn Pipeline:
1. **For Quantitative Features**:
   - Missing values imputed using median strategy
   - Features standardized to zero mean and unit variance

2. **For Nominal Features**:
   - One-hot encoding applied to handedness
   - Unknown categories handled gracefully with 'ignore' strategy

### Performance
- **Test Set Accuracy**: 0.5857 (58.57%)
- **Baseline Comparison**: Performs only 8.57 percentage points better than random guessing (50%)

### Evaluation
This baseline model's performance is relatively weak, suggesting significant room for improvement. The modest improvement over random guessing (only 8.57 percentage points) indicates that:
1. The selected features, particularly player handedness, may not be strong predictors of match outcomes
2. The simple decision tree structure may be insufficient to capture complex patterns in tennis match outcomes
3. Important predictive features from our exploratory analysis (like age and surface) are not yet incorporated

This baseline provides a clear starting point for improvement in our final model, where I can incorporate additional features identified in our exploratory analysis and potentially use more sophisticated modeling techniques.

## Final Model

To improve upon the baseline model's performance, I developed a more sophisticated approach using gradient boosting and carefully engineered features that capture the complex dynamics of tennis matches.

### Feature Engineering

I engineered several new features that capture different aspects of match dynamics:

1. **Player Differentials**
   - **Rank Difference** (`rank_diff`): Captures skill gap between players
   - **Age Difference** (`age_diff`): Reflects physical and experience disparities
   - **Ranking Points Difference** (`points_diff`): More granular measure of recent performance
   - **Elo Difference** (`elo_diff`): Dynamic rating system that updates after every match

2. **Tournament Context**
   - **Round Ordinal** (`round_ord`): Encoded match stages (R128→1 to F→7)
   - **Tournament Level** (`tourney_level_ord`): Encoded tournament prestige (Grand Slam→5 to Tour→0)
   - **Seasonal Features** (`month_sin`, `month_cos`): Cyclical encoding of tournament month

3. **Playing Conditions**
   - **Surface**: One-hot encoded court types (Hard, Clay, Grass)
   - **Player Handedness**: Encoded left/right-handed matchup dynamics

### Feature Preprocessing Pipeline

The model employs a sophisticated preprocessing pipeline with different strategies for each feature type:

1. **Numeric Features** (rankings, differences):
   - Missing value imputation
   - Standard scaling (zero mean, unit variance)

2. **Quantile Features** (points difference):
   - Missing value imputation
   - Quantile transformation to normal distribution

3. **Ordinal Features** (round, level, seasonal):
   - Passed through as already properly encoded

4. **Categorical Features** (surface, handedness):
   - One-hot encoding with unknown category handling

### Model Selection and Training

I chose the **Histogram-based Gradient Boosting Classifier** for several reasons:
1. Efficient handling of mixed data types
2. Ability to capture non-linear feature interactions
3. Built-in handling of missing values
4. Strong performance on tabular data

### Hyperparameter Tuning

Used 5-fold cross-validation with GridSearchCV to optimize:
- **Number of iterations**: Controls model complexity
- **Tree depth**: Balances detail vs. generalization
- **Learning rate**: Affects convergence speed
- **L2 regularization**: Controls overfitting

Best parameters found:
```python
{
    'max_iter': 100,
    'max_depth': 10,
    'learning_rate': 0.01,
    'l2_regularization': 0
}
```

### Performance Improvement

The final model achieved significant improvements over the baseline:
- **Baseline Model**: 58.57% accuracy
- **Final Model**: 63.70% accuracy
- **Improvement**: 5.13 percentage points

This improvement suggests that our engineered features successfully capture important match dynamics that the baseline model missed.

### Performance Visualization

<iframe
src="assets/confusion_matrix.html"
width="800"
height="600"
frameborder="0"
></iframe>

The confusion matrix shows balanced performance across both positive and negative predictions, indicating the model isn't biased toward either outcome.

### Feature Impact Analysis

The most influential features in predicting match outcomes were:
1. Ranking difference (capturing skill gap)
2. Elo difference (recent performance)
3. Tournament level (match importance)
4. Surface (playing conditions)

This aligns with tennis expertise, where ranking and recent form are typically the strongest predictors of match outcomes.

### Model Limitations

While the model shows good improvement, some limitations remain:
1. Doesn't capture player-specific surface preferences
2. Cannot account for injuries or fatigue
3. May not fully capture momentum from recent tournament performance

These limitations suggest potential areas for future model improvements, such as incorporating player-surface win rates or recent match history features.

---
*This project was created as part of EECS 398: Practical Data Science at the University of Michigan.*
