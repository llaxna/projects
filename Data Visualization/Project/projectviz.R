############################## Data Loading #####################################


# Load necessary libraries
library(dplyr)
library(tidyr)
library(readr)
library(readxl)
library(ggplot2)
library(car)
library(lmtest)
library(stats)
library(ggmosaic)
library(janitor)
library(kableExtra)
library(sf)
library(tmap)


df <- read_excel("/Users/yazeedkarajih/Downloads/MOH-SQW_Survey.xlsx")

print(names(df))

colnames(df) <- c("Gender", "Age", "City", "Gov_Assistance", "Education_Level", "Physical_Activity", "Breakfast_Frequency", "Whole_Grain_Consumption", "Dairy_Products_Consumption",
                  "Caffeine_Consumption", "Fruits_Consumption", "Vegetables_Consumption", "Fish_Consumption", "Food_Insecurity_Worry", "Unhealthy_Food_Lack", "Limited_Food_Variety", 
                  "Meal_Skipping", "Reduced_Food_Intake", "Household_Ran_Out_Of_Food", "Hunger_No_Eating", "Day_Without_Eating", "Bounce_Back_Quickly", "Hard_Time_During_Stress", 
                  "Quick_Recovery_From_Stress", "Difficulty_With_Recovery", "Ease_Through_Difficult_Times", "Slow_Recovery_From_Setbacks", "Stress_Negative_View", "Stress_Learning_Growth", 
                  "Stress_Depletes_Health", "Stress_Enhances_Performance", "Stress_Inhibits_Growth", "Stress_Improves_Health", "Stress_Debilitates_Performance", "Stress_Positive_View", 
                  "Upset_By_Unexpected", "Control_Over_Life", "Felt_Nervous_Stressed", "Confidence_Handling_Problems", "Things_Going_Your_Way", "Overwhelmed_By_Responsibilities",
                  "Angered_By_Outside_Control", "Overcome_By_Difficulties", "Past_Month_Feeling_Nervous", "Past_Month_Feeling_Hopeless")

print(names(df))

df$Age <- factor(df$Age, levels = c("18-22", "23-29", "30-39", "40-49", "50 and older"))
df$Education_Level <- factor(df$Education_Level, levels = c("High School", "Undergraduate", "Graduate"))
df$Gender <- as.factor(df$Gender)
df$Caffeine_Consumption <- as.factor(df$Caffeine_Consumption)

col_mode <- c("Gender", "Age", "City", "Gov_Assistance", "Education_Level", "Physical_Activity", "Breakfast_Frequency", "Whole_Grain_Consumption", "Dairy_Products_Consumption",
              "Caffeine_Consumption", "Fruits_Consumption", "Vegetables_Consumption", "Fish_Consumption", "Food_Insecurity_Worry", "Unhealthy_Food_Lack", "Limited_Food_Variety", 
              "Meal_Skipping", "Reduced_Food_Intake", "Household_Ran_Out_Of_Food", "Hunger_No_Eating", "Day_Without_Eating","Stress_Negative_View", "Stress_Learning_Growth", 
              "Stress_Depletes_Health", "Stress_Enhances_Performance", "Stress_Inhibits_Growth", "Stress_Improves_Health", "Stress_Debilitates_Performance", "Stress_Positive_View", 
              "Upset_By_Unexpected", "Control_Over_Life", "Felt_Nervous_Stressed", "Confidence_Handling_Problems", "Things_Going_Your_Way", "Overwhelmed_By_Responsibilities",
              "Angered_By_Outside_Control", "Overcome_By_Difficulties", "Past_Month_Feeling_Nervous", "Past_Month_Feeling_Hopeless")

null_values <- is.na(df)
summary_null <- colSums(null_values)
print(summary_null)

for (col in col_mode) {
  calculate_mode <- function(x) {
    unique_values <- unique(x)
    mode_value <- unique_values[which.max(tabulate(match(x, unique_values)))]
    return(mode_value)
  }
  mode_value <- calculate_mode(df[[col]])
  df[[col]][is.na(df[[col]])] <- mode_value
}

null_values <- is.na(df)
summary_null <- colSums(null_values)
print(summary_null)

gadm_file <- "/Users/yazeedkarajih/Downloads/gadm41_JOR_shp/gadm41_JOR_1.shp"  # Update with your actual path

jordan_admin <- st_read(gadm_file)

tmap_mode("view")  # Change to "plot" if you prefer a static map

city_counts <- df %>%
  dplyr::group_by(City) %>%
  dplyr::summarise(Count = n())

city_counts

values_data <- data.frame(
  NAME_1 = city_counts$City,  # Governorate names
  Value = city_counts$Count  # Corresponding values
)

jordan_admin <- jordan_admin %>%
  left_join(values_data, by = c("NAME_1"))


tmap_mode("view")  

# Plot the map, highlighting the governorates based on values
tm_shape(jordan_admin) +
  tm_borders() +
  tm_fill("Value", palette = c('lightgreen','orange'), title = "Counts") +  
  tm_text("NAME_1", size=0.7)+
  tm_layout(title = "Participant Counts")


############################## Resilience #####################################
Resilience <- c("Bounce_Back_Quickly", "Hard_Time_During_Stress", "Quick_Recovery_From_Stress", 
                "Difficulty_With_Recovery", "Ease_Through_Difficult_Times", "Slow_Recovery_From_Setbacks")

df$Hard_Time_During_Stress <- 6 - df$Hard_Time_During_Stress
df$Difficulty_With_Recovery <- 6 - df$Difficulty_With_Recovery
df$Slow_Recovery_From_Setbacks <- 6 - df$Slow_Recovery_From_Setbacks

for (col in Resilience) {
  unique_vals <- unique(df[[col]])
  print(unique_vals)
}

for (cols in Resilience) {
  for (col in cols) {
    median_val <- median(df[[col]], na.rm = TRUE)
    
    df[[col]] <- ifelse(is.na(df[[col]]), median_val, df[[col]])
  }
}

for (col in Resilience) {
  unique_vals <- unique(df[[col]])
  print(unique_vals)
}

#add column score for Resilience
col_name_res <- "Slow_Recovery_From_Setbacks" 

insert_position_res <- which(colnames(df) == col_name_res)
df <- cbind(df[, 1:insert_position_res], 'Resilience Score' = rowSums(df[Resilience], na.rm = TRUE), df[, (insert_position_res+1):ncol(df)])

summary_stats_res <- summary(df$'Resilience Score')
print(summary_stats_res)

low <- sum(df$'Resilience Score' >= 6 & df$'Resilience Score' <= 13)
normal <- sum(df$'Resilience Score' >= 14 & df$'Resilience Score' <= 21)
high <- sum(df$'Resilience Score' >= 22 & df$'Resilience Score' <= 30)

print(paste("Likely to have low Resilience:", low))
print(paste("Likely to have normal Resilience:", normal))
print(paste("Likely to have high Resilience:", high))

scores <- c("low", "normal", "high")
counts <- c(low, normal, high)

max_count <- max(counts)

# Create a bar plot
barplot(counts, names.arg = scores, col = "skyblue",
        main = "Distribution of Resilience Scores",
        xlab = "Resilience Score Categories",
        ylab = "Number of Participants",
        ylim = c(0, max_count + 50))

text(x = barplot(counts, names.arg = scores, col = "skyblue", main = "Distribution of Resilience Scores", xlab = "Resilience Score Categories", ylab = "Number of Participants", ylim = c(0, max_count + 50)),
     y = counts + 20, labels = counts, pos = 3, cex = 0.8)

#create histogram
dev.off()
par(mar = c(2, 2, 2, 2) + 0.1)

par(mfrow=c(2, 3)) 
for (col in Resilience) {
  hist(df[[col]], main = col, xlab = "Response", ylab = "Frequency")
}

dev.off()
hist(df$'Resilience Score', main = "Distribution of Resilience Scores", xlab = "Resilience Score", ylab = "Frequency", col = "skyblue")

lm_model <- lm(`Resilience Score` ~ Gender + Age + Education_Level + Physical_Activity + City, data = df)
summary(lm_model)

#Gender: Males tend to have higher Resilience scores compared to females, as indicated by the positive coefficient for GenderMale.
#Age: Older age groups, such as those aged 40 and older, tend to have higher Resilience scores compared to younger age groups.
#Education Level: None of the education level categories show a statistically significant effect on stress mindset scores, as indicated by the p-values above the significance level of 0.05.
#Physical Activity: Participants engaging in physical activity 5 times a week and more tend to have higher Resilience scores compared to those with less frequent physical activity.
#City: The city of residence does not appear to have a significant association with Resilience scores.

anova(lm_model)

#Gender, Age, Education Level, and Physical Activity have statistically significant effects on resilience scores (p < 0.05)
#City does not have a statistically significant effect on resilience scores, as its p-value is much higher than 0.05.

rmse <- sqrt(mean(lm_model$residuals^2))
rsquared <- summary(lm_model)$r.squared
adj_rsquared <- summary(lm_model)$adj.r.squared

print(paste("Root Mean Squared Error (RMSE):", rmse))
print(paste("R-squared:", rsquared))
print(paste("Adjusted R-squared:", adj_rsquared))

############################## stress_mindset #####################################
stress_mindset <- c('Stress_Learning_Growth', 'Stress_Enhances_Performance', 'Stress_Improves_Health', 
                    'Stress_Positive_View', 'Stress_Negative_View', 'Stress_Depletes_Health', 
                    'Stress_Inhibits_Growth', 'Stress_Debilitates_Performance')

# all responses to lowercase 
for (col in stress_mindset) {
  df[[col]] <- tolower(df[[col]])
}

for (col in stress_mindset) {
  unique_vals <- unique(df[[col]])
  print(unique_vals)
}    

stress_positive <- c("strongly disagree" = 0, "disagree" = 1, "neither agree nor disagree" = 2, "agree" = 3, "strongly agree" = 4)
stress_negative <- c("strongly disagree" = 4, "disagree" = 3, "neither agree nor disagree" = 2, "agree" = 1, "strongly agree" = 0)

stress_pos <- c('Stress_Learning_Growth', 'Stress_Enhances_Performance', 'Stress_Improves_Health', 'Stress_Positive_View')
stress_neg <- c('Stress_Negative_View', 'Stress_Depletes_Health', 'Stress_Inhibits_Growth', 'Stress_Debilitates_Performance') 

#convert from categorical to numerical
for (col in stress_pos) {
  df[[col]] <- stress_positive[df[[col]]]
}

for (col in stress_neg) {
  df[[col]] <- stress_negative[df[[col]]]
}

for (col in stress_mindset) {
  unique_vals <- unique(df[[col]])
  print(unique_vals)
}  

#add column score for stress_mindset
col_name <- 'Stress_Positive_View'  

insert_position <- which(colnames(df) == col_name)
df <- cbind(df[, 1:insert_position], 'Stress Mindset Score' = rowSums(df[stress_mindset], na.rm = TRUE), df[, (insert_position+1):ncol(df)])

summary_stats_sm <- summary(df$'Stress Mindset Score')
print(summary_stats_sm)

debilitating <- sum(df$'Stress Mindset Score' >= 0 & df$'Stress Mindset Score' <= 10)
moderate <- sum(df$'Stress Mindset Score' >= 11 & df$'Stress Mindset Score' <= 21)
enhancing <- sum(df$'Stress Mindset Score' >= 22 & df$'Stress Mindset Score' <= 32)

print(paste("Likely to have a debilitating effect:", debilitating))
print(paste("Likely to have a moderate effect:", moderate))
print(paste("Likely to have an enhancing effect:", enhancing))

scores <- c("Debilitating", "Moderate", "Enhancing")
counts <- c(debilitating, moderate, enhancing)

max_count <- max(counts)

# Create a bar plot
dev.off()
barplot(counts, names.arg = scores, col = "skyblue",
        main = "Distribution of Stress Mindset Scores",
        xlab = "Stress Mindset Score Categories",
        ylab = "Number of Participants",
        ylim = c(0, max_count + 50))

text(x = barplot(counts, names.arg = scores, col = "skyblue", main = "Distribution of Stress Mindset Scores", xlab = "Stress Mindset Score Categories", ylab = "Number of Participants", ylim = c(0, max_count + 50)),
     y = counts + 20, labels = counts, pos = 3, cex = 0.8)

#create histogram
dev.off()
par(mar = c(5, 4, 4, 2) + 0.1)

par(mfrow=c(2, 4)) 
for (col in stress_mindset) {
  hist(df[[col]], main = col, xlab = "Response", ylab = "Frequency")
}

dev.off()
hist(df$'Stress Mindset Score', main = "Distribution of Stress Mindset Scores", xlab = "Stress Mindset Score", ylab = "Frequency", col = "skyblue")

lm_model <- lm(`Stress Mindset Score` ~ Gender + Age + Education_Level + Physical_Activity + City, data = df)
summary(lm_model)

#Gender: Males tend to have higher stress mindset scores compared to females, as indicated by the positive coefficient for GenderMale.
#Age: Older age groups, such as those aged 50 and older, tend to have lower stress mindset scores compared to younger age groups.
#Education Level: those with an undergraduate degree have higher stress mindset scores.
#Physical Activity: Participants engaging in physical activity 4 times a week and more tend to have higher stress mindset scores compared to those with less frequent physical activity.
#City: The city of residence does not appear to have a significant association with stress mindset scores.

anova(lm_model)

#Both the Gender and Age predictors are highly significant (p < 0.05), indicating that they are important predictors of stress mindset scores. 
#Physical Activity (PW) also shows significance, indicating its importance in predicting stress mindset scores.

rmse <- sqrt(mean(lm_model$residuals^2))
rsquared <- summary(lm_model)$r.squared
adj_rsquared <- summary(lm_model)$adj.r.squared

print(paste("Root Mean Squared Error (RMSE):", rmse))
print(paste("R-squared:", rsquared))
print(paste("Adjusted R-squared:", adj_rsquared))


################################## mental_distress ################################## 
mental_distress <- c( "Upset_By_Unexpected", "Control_Over_Life", "Felt_Nervous_Stressed", "Confidence_Handling_Problems", "Things_Going_Your_Way", 
                      "Overwhelmed_By_Responsibilities","Angered_By_Outside_Control", "Overcome_By_Difficulties", "Past_Month_Feeling_Nervous", "Past_Month_Feeling_Hopeless")

for (col in mental_distress) {
  unique_vals <- unique(df[[col]])
  print(unique_vals)
}

# all responses to lowercase 
for (col in mental_distress) {
  df[[col]] <- tolower(df[[col]])
}

mental_distress_num <- c("never" = 1, "almost never" = 2, "sometimes" = 3, "fairly often" = 4, "very often" = 5, "alfairly often never"=2)

#convert from categorical to numerical
for (col in mental_distress) {
  df[[col]] <- mental_distress_num[df[[col]]]
}

for (col in mental_distress) {
  unique_vals <- unique(df[[col]])
  print(unique_vals)
}

#add column score for mental_distress
df$'Mental Distress Score' <- rowSums(df[mental_distress], na.rm = TRUE)

summary_stats_md <- summary(df$'Mental Distress Score')
print(summary_stats_md)

well <- sum(df$'Mental Distress Score' >= 10 & df$'Mental Distress Score' <= 19)
moderate <- sum(df$'Mental Distress Score' >= 20 & df$'Mental Distress Score' <= 29)
severe <- sum(df$'Mental Distress Score' >= 30 & df$'Mental Distress Score' <= 50)

print(paste("Likely to be well:", well))
print(paste("Likely to have a moderate disorder:", moderate))
print(paste("Likely to have a severe disorder:", severe))

scores <- c("Well", "Moderate", "Severe")
counts <- c(well, moderate, severe)

max_count <- max(counts)

# Create a bar plot
dev.off()
barplot(counts, names.arg = scores, col = "skyblue",
        main = "Distribution of Mental Distress Scores",
        xlab = "Mental Distress Score Categories",
        ylab = "Number of Participants",
        ylim = c(0, max_count + 50))

text(x = barplot(counts, names.arg = scores, col = "skyblue", main = "Distribution of Mental Distress Scores", xlab = "Mental Distress Score Categories", ylab = "Number of Participants", ylim = c(0, max_count + 50)),
     y = counts + 20, labels = counts, pos = 3, cex = 0.8)

dev.off()
hist(df$'Mental Distress Score', main = "Distribution of Mental Distress Scores", xlab = "Mental Distress Score", ylab = "Frequency", col = "skyblue")

################################## food_security ################################## 
food_security <- c( "Food_Insecurity_Worry", "Unhealthy_Food_Lack", "Limited_Food_Variety", "Meal_Skipping", 
                    "Reduced_Food_Intake", "Household_Ran_Out_Of_Food", "Hunger_No_Eating", "Day_Without_Eating")

for (col in food_security) {
  unique_vals <- unique(df[[col]])
  print(unique_vals)
}

# all responses to lowercase 
for (col in food_security) {
  df[[col]] <- tolower(df[[col]])
}

food_security_num <- c("yes" = 1, "no" = 0, "sometimes true" = 1)

#convert from categorical to numerical
for (col in food_security) {
  df[[col]] <- food_security_num[df[[col]]]
}


for (col in food_security) {
  unique_vals <- unique(df[[col]])
  print(unique_vals)
}


#add column score for food_security
col_name <- 'Day_Without_Eating'  

insert_position <- which(colnames(df) == col_name)
df <- cbind(df[, 1:insert_position], 'Food Security Score' = rowSums(df[food_security], na.rm = TRUE), df[, (insert_position+1):ncol(df)])

summary_stats_fs <- summary(df$'Food Security Score')
print(summary_stats_fs)

high <- sum(df$'Food Security Score' >= 0 & df$'Food Security Score' <= 1)
low <- sum(df$'Food Security Score' >= 2 & df$'Food Security Score' <= 4)
very_low <- sum(df$'Food Security Score' >= 4 & df$'Food Security Score' <= 8)

print(paste("High food security:", high))
print(paste("Low food security:", low))
print(paste("Very low food security:", very_low))

scores <- c("high", "low", "very low")
counts <- c(high, low, very_low)

max_count <- max(counts)

# Create a bar plot
dev.off()
barplot(counts, names.arg = scores, col = "skyblue",
        main = "Distribution of Food Security Scores",
        xlab = "Food Security Score Categories",
        ylab = "Number of Participants",
        ylim = c(0, max_count + 50))

text(x = barplot(counts, names.arg = scores, col = "skyblue", main = "Distribution of Food Security Scores", xlab = "Food Security Score Categories", ylab = "Number of Participants", ylim = c(0, max_count + 50)),
     y = counts + 20, labels = counts, pos = 3, cex = 0.8)

# Create histogram with sky blue fill and bars sticking together
ggplot(df, aes(x = df$`Food Security Score`)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black", position = "identity") +
  labs(title = "Histogram of Food Security",
       x = "Food Security Score",
       y = "Frequency") +
  theme_minimal()


################# Drop Factor Columns #########################
cols_to_drop <- c(Resilience, stress_mindset, mental_distress, food_security)
df <- df %>% select(-all_of(cols_to_drop))

############################## Caffeine Consumption vs Stress Mindset Score #####################################

# Ensure Caffeine_Consumption is categorical
df$Caffeine_Consumption <- factor(df$Caffeine_Consumption, levels = unique(c("Never","1 time", "2 times", "3 times", "4 times","5 times or more")))

# Verify the columns and data types
print(names(df))

# Check for unique values in Caffeine_Consumption
unique(df$Caffeine_Consumption)

# Q-Q plot for each group
#Values are normally distrubuted
qqPlot(lm(df$'Stress Mindset Score' ~ df$Caffeine_Consumption), xlab = "Theoretical Quantiles",
       ylab = "Sample Quantiles")

leveneTest(df$'Stress Mindset Score' ~ df$Caffeine_Consumption)

anova_result <- aov(df$'Stress Mindset Score' ~ df$Caffeine_Consumption)
summary(anova_result)

ggplot(df, aes(x=Caffeine_Consumption, y= df$'Stress Mindset Score', fill = Caffeine_Consumption))+
  geom_boxplot()+
  scale_fill_brewer(palette = 'Set3')+
  labs(title = 'Stress Mindset vs Consumption',
       y='Stress Mindset',
       fill='Caff')

ggplot(df, aes(x = df$'Stress Mindset Score', fill = Caffeine_Consumption)) +
  geom_histogram(binwidth = 5, color = "black", alpha = 0.7,  stat = "count") +
  labs(title = "Histogram of Stress Mindset Score",
       fill = 'Caffeine Consumption',
       x = "Stress Mindset Score",
       y = "Frequency") +
  facet_wrap(~df$Caffeine_Consumption)+
  scale_fill_manual(values = facet_colors)+
  theme_minimal()+
  
  theme(axis.text.x = element_blank())

############################## Gender vs Resilience Score #####################################

category1 <- "Female"
category2 <- "Male"
filtered_data <- df %>% filter(Gender %in% c(category1, category2))



ggplot(filtered_data, aes(x = factor(Gender), y = filtered_data$'Resilience Score', fill=Gender) )+
  scale_fill_manual(values = c("pink", "skyblue")) +
  geom_boxplot() +
  labs(x = "Gender", y = "Resilience Score", 
       title = "Comparison of Resilience Scores by Gender") +
  theme_minimal()


ggplot(filtered_data, aes(x = filtered_data$'Resilience Score', fill = Gender)) +
  geom_histogram(color = "black", alpha = 0.7,  stat = "count") +
  labs(title = "Histogram of Resilience Scores",
       x = "Stress Mindset Score",
       y = "Frequency") +
  facet_wrap(~filtered_data$Gender)+
  scale_fill_manual(values = c('pink','skyblue'))+
  theme_minimal()+
  theme(axis.text.x = element_blank())

qqPlot(lm(filtered_data$'Resilience Score' ~ filtered_data$Gender))

leveneTest(filtered_data$'Resilience Score' ~ filtered_data$Gender)

anova_result <- aov(filtered_data$'Resilience Score' ~ filtered_data$Gender)
summary(anova_result)

model <- lm(df$`Resilience Score` ~ Gender + City + Gov_Assistance + Age, data = df)
summary(model)

anova_result <- aov(filtered_data$'Resilience Score' ~ filtered_data$Gender + filtered_data$Age)
summary(anova_result)

names(filtered_data)[names(filtered_data) == 'Resilience Score'] <- 'Resilience_Score'

model <- lm(Resilience_Score ~ Gender, data = filtered_data)

summary(model)

perform_regression <- function(stratum) {
  model <- lm(Resilience_Score ~ Gender, data = stratum)
  summary(model)
}

strata <- split(filtered_data, filtered_data$Age)

results <- lapply(strata, perform_regression)

results ####### Age has an interaction effect on the relationship between Genders and Resilience depends on the age group.

mean_data <- aggregate(Resilience_Score ~ Gender, data = filtered_data, FUN = median)

mean_data

ggplot(filtered_data, aes(x = filtered_data$'Resilience_Score', fill = Gender)) +
  geom_density(alpha = 0.5) + 
  geom_vline(data = mean_data, aes(xintercept = Resilience_Score, color = Gender),
             linetype = "dashed", size = 1)+
  labs(title = "Smoothed Density Plot of Resilience Score by Category",
       x = "Resilience Score",
       y = "Density") +
  scale_fill_manual(values = c('pink','skyblue'))+
  theme_minimal()

ggplot(filtered_data, aes(x = factor(Gender), y = filtered_data$'Resilience_Score', fill=Gender) )+
  scale_fill_manual(values = c("pink", "skyblue")) +
  geom_boxplot() +
  labs(x = "Gender", y = "Resilience Score", 
       title = "Comparison of Resilience Scores by Gender") +
  facet_wrap(~Age)+
  theme_minimal()

############################## Age vs Mental Distress Score #####################################
  
category1 <- "Female"

category2 <- "Male"
filtered_data <- df %>% filter(Gender %in% c(category1, category2))

model <- lm(filtered_data$`Mental Distress Score` ~ Gender + City + Gov_Assistance + Age, data = filtered_data)
summary(model)

names(df)[names(df) == 'Mental Distress Score'] <- 'Mental_Distress'

names(filtered_data)[names(filtered_data) == 'Mental Distress Score'] <- 'Mental_Distress'


perform_regression <- function(stratum) {
  model <- lm(Mental_Distress ~ Age, data = stratum)
  summary(model)
}

strata <- split(df, df$Gender)

results <- lapply(strata, perform_regression)

results ######## Interaction Effect of Gender on Age and Mental Distress


ggplot(filtered_data, aes(x = Age, y = Mental_Distress, fill = Gender)) +
  geom_boxplot() +
  scale_fill_manual(values = c('pink','skyblue'))+
  labs(title = "Interaction Effect of Gender on Mental Distress and Age Group") 

  
ggplot(filtered_data, aes(x=Age, y= filtered_data$'Mental_Distress', fill = Age))+
  geom_boxplot()+
  scale_fill_brewer(palette = 'Set3')+
  labs(title = 'Mental Distress vs Age',
       y='Mental Distress')

  
ggplot(filtered_data, aes(x = filtered_data$'Mental_Distress', fill = Age)) +
  geom_density(alpha = 0.5) +  # Add smoothed density curves with transparency
  labs(title = "Mental Distress Score vs Age",
       x = "Mental Distress Score",
       y = "Density") +
  scale_fill_manual(values = c('lightgreen','white','white','white','mediumorchid1' ))+
  theme_minimal()

category1 <- "18-22"
category2 <- "50 and older"
filtered_data <- filtered_data %>% filter(Age %in% c(category1, category2))

mean_data <- aggregate(Mental_Distress ~ Age, data = filtered_data, FUN = median)

mean_data

ggplot(filtered_data, aes(x = Mental_Distress, fill = Age)) +
  geom_density(alpha = 0.5) +
  geom_vline(data = mean_data, aes(xintercept = Mental_Distress, color = Age),
             linetype = "dashed", size = 1) +
  labs(title = "Mental Distress Score vs Age",
       x = "Mental Distress Score",
       y = "Density") +
  scale_fill_manual(values = c('18-22'='lightgreen','50 and older'='mediumorchid1'))+
  scale_color_manual(values = c('18-22'='lightgreen','50 and older'='mediumorchid1')) +
  theme_minimal()

  
qqPlot(lm(filtered_data$'Mental_Distress' ~ filtered_data$Age))

leveneTest(filtered_data$'Mental_Distress' ~ filtered_data$Age)

oneway.test(filtered_data$'Mental_Distress' ~ filtered_data$Age, var.equal = FALSE)

names(df)[names(df) == 'Mental_Distress'] <- 'Mental Distress Score'

############################## Food Security vs Government Assistance #####################################
df <- mutate(df, 'FC' = case_when(
  df$'Food Security Score' %in% 0:1 ~ "high",
  df$'Food Security Score' %in% 2:4 ~ "low",
  df$'Food Security Score' %in% 5:8 ~ "very low",
  TRUE ~ NA  # Add NA for any other values (optional)
))


ggplot(data = df) +
  geom_mosaic(aes(x = product(FC,Gov_Assistance), fill = FC))+
  labs(title = "Food Security vs Government Assistance",
       x = "Government Assistance",
       y = "Food Security",
       fill="Food Security")

totals <- df %>%
  tabyl(FC, Gov_Assistance) %>%
  adorn_totals(where = c("row", "col"))

totals %>%
  kable(caption = "Food Security VS Government Assistance", booktabs = TRUE)

 thenchisq_test <- chisq.test(totals)

print(chisq_test)


############################ Scores Analysis ###################################

model <- lm(df$`Stress Mindset Score` ~ df$`Resilience Score` + df$`Food Security Score` + df$`Mental Distress Score`, data = df)
summary(model)

model <- lm(df$`Resilience Score` ~ df$`Stress Mindset Score` + df$`Food Security Score` + df$`Mental Distress Score`, data = df)
summary(model)

model <- lm(df$`Food Security Score` ~ df$`Stress Mindset Score` + df$`Resilience Score` + df$`Mental Distress Score`, data = df)
summary(model)


model <- lm(df$`Mental Distress Score` ~ df$`Stress Mindset Score` + df$`Resilience Score` + df$`Food Security Score`, data = df)
summary(model)

ggplot(data = df, aes(x = df$`Food Security Score`, y = df$`Mental Distress Score`)) +
  geom_point()+
  labs(title = "Food Security vs Government Assistance",
       x = "Government Assistance",
       y = "Food Security",
       fill="Food Security")

ggplot(data = df, aes(x = df$`Food Security Score`, y = df$`Mental Distress Score`)) +
  geom_jitter(width = 0.5, height = 0.2) +
  geom_smooth(method='lm')+
  labs(title = "Food Security vs Government Assistance",
       x = "Food Security",
       y = "Mental Distress",
       fill="Food Security")

ggplot(data = df, aes(x = df$`Food Security Score`, y = df$`Mental Distress Score`)) +
  geom_jitter(width = 0.5, height = 0.2) +
  geom_smooth(method='lm')+
  facet_wrap(~City)+
  labs(title = "Food Security vs Government Assistance",
       x = "Food Security",
       y = "Mental Distress",
       fill="Food Security")
##############################################
ggplot(data = df, aes(x = df$`Resilience Score`, y = df$`Mental Distress Score`)) +
  geom_point()+
  labs(title = "Resilience vs Mental Distress",
       x = "Government Assistance",
       y = "Food Security",
       fill="Food Security")

ggplot(data = df, aes(x = df$`Resilience Score`, y = df$`Mental Distress Score`)) +
  geom_jitter(width = 0.5, height = 0.2) +
  geom_smooth(method='lm')+
  labs(title = "Resilience vs Mental Distress",
       x = "Food Security",
       y = "Mental Distress",
       fill="Food Security")

category1 <- "Female"
category2 <- "Male"
filtered_data <- df %>% filter(Gender %in% c(category1, category2))

ggplot(data = filtered_data, aes(x = filtered_data$`Resilience Score`, y = filtered_data$`Mental Distress Score`, col = Gender)) +
  geom_jitter(width = 0.5, height = 0.2) +
  geom_smooth(method='lm')+
  facet_wrap(~Gender)+
  labs(title = "Resilience vs Mental Distress",
       x = "Resilience",
       y = "Mental Distress",
       fill="Food Security")

model1 <- lm(df$`Mental Distress Score` ~ df$`Resilience Score`)
summary(model1)

model2 <- lm(df$`Mental Distress Score` ~ df$`Resilience Score` + df$Gender)
summary(model2)

coef(model1)
coef(model2)
# Gender is not a confounder.

