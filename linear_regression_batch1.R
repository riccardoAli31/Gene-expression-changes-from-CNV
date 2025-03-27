library(ggplot2)
library(GGally)
library(Metrics)  # For calculating MSE

df_test1 <- read.csv("/Users/gajaj/OneDrive/Desktop/yX_test1.csv")
df_val1 <- read.csv("/Users/gajaj/OneDrive/Desktop/yX_val1.csv")
df_train1 <- read.csv("/Users/gajaj/OneDrive/Desktop/yX_train1.csv")

# Convert response vector into a dataframe
y_train1 <- as.vector(df_train1$expression)
y_test1 <- as.vector(df_test1$expression)
y_val1 <- as.vector(df_val1$expression)

# LINEAR MODEL
model1 <- lm(expression ~ ., data = df_train1)
summary(model1)

# Forward AIC selection
#out1 <- lm(expression ~1, data=df_train1)
#forw1 <- step(out1, direction = 'forward', scope ~ AAA + AAT + AAG + AAC + ATA + ATT + ATG + ATC + AGA + AGT + AGG + AGC + ACA + ACT + ACG + ACC + TAA + TAT + TAG + TAC + TTA + TTT + TTG + TTC + TGA + TGT + TGG + TGC + TCA + TCT + TCG + TCC + GAA + GAT + GAG + GAC + GTA + GTT + GTG + GTC + GGA + GGT + GGG + GGC + GCA + GCT + GCG + GCC + CAA + CAT + CAG + CAC + CTA + CTT + CTG + CTC + CGA + CGT + CGG + CGC + CCA + CCT + CCG + CCC + cnv_loss_avg + cnv_gain_avg + open_chromatin_avg, data=df_ltrain1, trace=0)
#summary(forw1)


# Make predictions on the test set
y_pred1_test <- predict(model1, newdata = df_test1)
y_pred1_val <- predict(model1, newdata = df_val1)

# Calculate performance metrics
mse_test1 <- mse(y_test1, y_pred1_test)
r2_test1 <- 1 - sum((y_test1 - y_pred1_test)^2) / sum((y_test1 - mean(y_test1))^2)
mse_val1 <- mse(y_val1, y_pred1_val)
r2_val1 <- 1 - sum((y_val1 - y_pred1_val)^2) / sum((y_val1 - mean(y_val1))^2)

# Visualize predictions vs actual values
ggplot(data.frame(actual = y_val1, predicted = y_pred1_val), aes(x = actual, y = predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Predicted vs Actual Gene Expression",
       x = "Actual Gene Expression",
       y = "Predicted Gene Expression") +
  theme_minimal()

# Compute residuals
residuals_val1 <- df_val1$expression - y_pred1_val
residuals_test1 <- df_test1$expression - y_pred1_test

# Visualize Residuals Plot
ggplot(data.frame(predicted = y_pred1_val, residuals = residuals_val1), aes(x = predicted, y = residuals)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residuals vs Predicted Values",
       x = "Predicted Gene Expression",
       y = "Residuals") +
  theme_minimal()

qqnorm(residuals_val1)                    # QQ-plot
qqline(residuals_val1, col = "red")        # Reference line

############## LOG TRANSFORMATION ##############################################

# Transform the expression values
df_train1$log_expression <- log(df_train1$expression)
df_val1$log_expression <- log(df_val1$expression)
df_test1$log_expression <- log(df_test1$expression)

# Fit the linear model to the log_expression response
model_log <- lm(log_expression ~ AAA + AAT + AAG + AAC + ATA + ATT + ATG + ATC + AGA + AGT + AGG + AGC + ACA + ACT + ACG + ACC + TAA + TAT + TAG + TAC + TTA + TTT + TTG + TTC + TGA + TGT + TGG + TGC + TCA + TCT + TCG + TCC + GAA + GAT + GAG + GAC + GTA + GTT + GTG + GTC + GGA + GGT + GGG + GGC + GCA + GCT + GCG + GCC + CAA + CAT + CAG + CAC + CTA + CTT + CTG + CTC + CGA + CGT + CGG + CGC + CCA + CCT + CCG + CCC + cnv_loss_avg + cnv_gain_avg + open_chromatin_avg, data=df_train1)
summary(model_log)

# MSE for validation
mse_log1_val <- mse(df_val1$log_expression, predict(model_log, newdata = df_val1))

# Residual plot (on validation)
residuals_log1_val <- df_val1$log_expression - predict(model_log, newdata = df_val1)

r2_log1_val <- 1 - sum((residuals_log1_val)^2) / sum((df_val1$log_expression - mean(df_val1$log_expression))^2)

# MSE for test
mse_log1_test <- mse(df_test1$log_expression, predict(model_log, newdata = df_test1))

# Residual plot (on validation)
residuals_log1_test <- df_test1$log_expression - predict(model_log, newdata = df_test1)

r2_log1_test <- 1 - sum((residuals_log1_test)^2) / sum((df_test1$log_expression - mean(df_test1$log_expression))^2)

ggplot(data.frame(predicted = predict(model_log, newdata = df_val1), residuals = residuals_log1_val),
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


