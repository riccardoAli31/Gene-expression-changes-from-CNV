library(ggplot2)
library(GGally)
library(Metrics)  # For calculating MSE

df_test2 <- read.csv("/Users/gajaj/OneDrive/Desktop/yX_test2.csv")
df_train2 <- read.csv("/Users/gajaj/OneDrive/Desktop/yX_train2.csv")
df_val2 <- read.csv("/Users/gajaj/OneDrive/Desktop/yX_val2.csv")

# Convert response vector into a dataframe
y_train2 <- as.vector(df_train2$expression)
y_test2 <- as.vector(df_test2$expression)
y_val2 <- as.vector(df_val2$expression)

# Fit the linear model
model2 <- lm(expression ~ ., data = df_train2)
summary(model2)


# Make predictions on the test set
y_pred2_test <- predict(model2, newdata = df_test2)
y_pred2_val <- predict(model2, newdata = df_val2)

# Calculate performance metrics
mse_test2 <- mse(y_test2, y_pred2_test)
r2_test2 <- 1 - sum((y_test2 - y_pred2_test)^2) / sum((y_test2 - mean(y_test2))^2)

mse_val2 <- mse(y_val2, y_pred2_val)
r2_val2 <- 1 - sum((y_val2 - y_pred2_val)^2) / sum((y_val2 - mean(y_val2))^2)


# Visualize predictions vs actual values
ggplot(data.frame(actual = y_test2, predicted = y_pred2_test), aes(x = actual, y = predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Predicted vs Actual Gene Expression",
       x = "Actual Gene Expression",
       y = "Predicted Gene Expression") +
  theme_minimal()

# Compute residuals
residuals_val2 <- df_val2$expression - y_pred2_val
residuals_test2 <- df_test2$expression - y_pred2_test

# Visualize Residuals Plot
ggplot(data.frame(predicted = y_pred2_val, residuals = residuals_val2), aes(x = predicted, y = residuals)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residuals vs Predicted Values",
       x = "Predicted Gene Expression",
       y = "Residuals") +
  theme_minimal()

############## LOG TRANSFORMATION ##############################################

# Transform the expression values
df_train2$log_expression <- log(df_train2$expression)
df_val2$log_expression <- log(df_val2$expression)
df_test2$log_expression <- log(df_test2$expression)

# Fit the linear model to the log_expression response
model_log2 <- lm(log_expression ~ AAA + AAT + AAG + AAC + ATA + ATT + ATG + ATC + AGA + AGT + AGG + AGC + ACA + ACT + ACG + ACC + TAA + TAT + TAG + TAC + TTA + TTT + TTG + TTC + TGA + TGT + TGG + TGC + TCA + TCT + TCG + TCC + GAA + GAT + GAG + GAC + GTA + GTT + GTG + GTC + GGA + GGT + GGG + GGC + GCA + GCT + GCG + GCC + CAA + CAT + CAG + CAC + CTA + CTT + CTG + CTC + CGA + CGT + CGG + CGC + CCA + CCT + CCG + CCC + cnv_loss_avg + cnv_gain_avg + open_chromatin_avg, data=df_train2)
summary(model_log2)

# MSE for validation
mse_log2_val <- mse(df_val2$log_expression, predict(model_log2, newdata = df_val2))

# Residual plot (on validation)
residuals_log2_val <- df_val2$log_expression - predict(model_log2, newdata = df_val2)

# R2 val
r2_log2_val <- 1 - sum((residuals_log2_val)^2) / sum((df_val2$log_expression - mean(df_val2$log_expression))^2)

# MSE for test
mse_log2_test <- mse(df_test2$log_expression, predict(model_log2, newdata = df_test2))

# Residual plot (on test)
residuals_log2_test <- df_test2$log_expression - predict(model_log2, newdata = df_test2)

# R2 test
r2_log2_test <- 1 - sum((residuals_log2_test)^2) / sum((df_test2$log_expression - mean(df_test2$log_expression))^2)


ggplot(data.frame(predicted = predict(model_log2, newdata = df_val2), residuals = residuals_log2_val),
       aes(x = predicted, y = residuals)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residuals vs Predicted Values (Log Model)",
       x = "Predicted Log Gene Expression",
       y = "Residuals") +
  theme_minimal()
