library(ggplot2)
library(GGally)
library(Metrics)  # For calculating MSE

df_test1_sc <- read.csv("/Users/gajaj/OneDrive/Desktop/yX_test1_subclone.csv")
df_train1_sc <- read.csv("/Users/gajaj/OneDrive/Desktop/yX_train1_subclone.csv")
df_val1_sc <- read.csv("/Users/gajaj/OneDrive/Desktop/yX_val1_subclone.csv")

# Convert response vector into a dataframe
y_train1_sc <- as.vector(df_train1_sc$expression)
y_test1_sc <- as.vector(df_test1_sc$expression)
y_val1_sc <- as.vector(df_val1_sc$expression)

# LINEAR MODEL
model1_sc <- lm(expression ~ ., data = df_train1_sc)
summary(model1_sc)


# Make predictions
y_pred1_test_sc <- predict(model1_sc, newdata = df_test1_sc)
y_pred1_val_sc <- predict(model1_sc, newdata = df_val1_sc)

# Calculate performance metrics
mse_test1_sc <- mse(y_test1_sc, y_pred1_test_sc)
r2_test1_sc <- 1 - sum((y_test1_sc - y_pred1_test_sc)^2) / sum((y_test1_sc - mean(y_test1_sc))^2)

mse_val1_sc <- mse(y_val1_sc, y_pred1_val_sc)
r2_val1_sc <- 1 - sum((y_val1_sc - y_pred1_val_sc)^2) / sum((y_val1_sc - mean(y_val1_sc))^2)


# Visualize predictions vs actual values
ggplot(data.frame(actual = y_test1_sc, predicted = y_pred1_test_sc), aes(x = actual, y = predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Predicted vs Actual Gene Expression",
       x = "Actual Gene Expression",
       y = "Predicted Gene Expression") +
  theme_minimal()

# Compute residuals
residuals_test_sc <- df_test1_sc$expression - y_pred1_test_sc

# Visualize Residuals Plot
ggplot(data.frame(predicted = y_pred1_test_sc, residuals = residuals_test_sc), aes(x = predicted, y = residuals)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residuals vs Predicted Values",
       x = "Predicted Gene Expression",
       y = "Residuals") +
  theme_minimal()

############## LOG TRANSFORMATION ##############################################

# Transform the expression values
df_train1_sc$log_expression <- log(df_train1_sc$expression)
df_test1_sc$log_expression <- log(df_test1_sc$expression)

# Fit the linear model to the log_expression response
model_log_sc <- lm(log_expression ~ AAA + AAT + AAG + AAC + ATA + ATT + ATG + ATC + AGA + AGT + AGG + AGC + ACA + ACT + ACG + ACC + TAA + TAT + TAG + TAC + TTA + TTT + TTG + TTC + TGA + TGT + TGG + TGC + TCA + TCT + TCG + TCC + GAA + GAT + GAG + GAC + GTA + GTT + GTG + GTC + GGA + GGT + GGG + GGC + GCA + GCT + GCG + GCC + CAA + CAT + CAG + CAC + CTA + CTT + CTG + CTC + CGA + CGT + CGG + CGC + CCA + CCT + CCG + CCC + cnv_loss_avg + cnv_gain_avg + open_chromatin_avg, data=df_train1_sc)
summary(model_log_sc)

# MSE for validation
mse_log_sc <- mse(df_test1_sc$log_expression, predict(model_log_sc, newdata = df_test1_sc))

# Residual plot (on validation)
residuals_log_sc <- df_test1_sc$log_expression - predict(model_log_sc, newdata = df_test1_sc)

r2_log1_sc <- 1 - sum((residuals_log_sc)^2) / sum((df_test1_sc$log_expression - mean(df_test1_sc$log_expression))^2)

ggplot(data.frame(predicted = predict(model_log_sc, newdata = df_test1_sc), residuals = residuals_log_sc),
       aes(x = predicted, y = residuals)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residuals vs Predicted Values (Log Model)",
       x = "Predicted Log Gene Expression",
       y = "Residuals") +
  theme_minimal()

############ POLYNOMIAL REGRESSION ######################################

predictor_vars <- colnames(df_train1)[!colnames(df_train1) %in% c("expression", "log_expression")]

df_train1_poly <- df_train1
df_val1_poly <- df_val1

for (var in predictor_vars) {
  df_train1_poly[[paste0(var, "_cube")]] <- df_train1[[var]]^2
  df_val1_poly[[paste0(var, "_cube")]] <- df_val1[[var]]^2  # Apply to validation set
}

model_poly <- lm(log_expression ~ ., data = df_train1_poly)
summary(model_poly)

y_pred_poly <- predict(model_poly, newdata = df_val1_poly)
residuals_poly <- df_val1$log_expression - y_pred_poly

ggplot(data.frame(predicted = y_pred_poly, residuals = residuals_poly),
       aes(x = predicted, y = residuals)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residuals vs Predicted Values (Quibic Model)",
       x = "Predicted Log Gene Expression",
       y = "Residuals") +
  theme_minimal()


