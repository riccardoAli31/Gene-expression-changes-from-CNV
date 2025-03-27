library(ggplot2)
library(GGally)
library(Metrics)  # For calculating MSE

df_train2_sub <- read.csv("/Users/gajaj/OneDrive/Desktop/yX_train2_sub.csv")

# Convert response vector into a dataframe
y_train2_sub <- as.vector(df_train2_sub$expression)

# Fit the linear model
model2_sub <- lm(expression ~ ., data = df_train2_sub)
summary(model2_sub)

# Forward AIC selection
out1 <- lm(expression ~1, data=df_learn1)
forw1 <- step(out1, direction = 'forward', scope ~ AAA + AAT + AAG + AAC + ATA + ATT + ATG + ATC + AGA + AGT + AGG + AGC + ACA + ACT + ACG + ACC + TAA + TAT + TAG + TAC + TTA + TTT + TTG + TTC + TGA + TGT + TGG + TGC + TCA + TCT + TCG + TCC + GAA + GAT + GAG + GAC + GTA + GTT + GTG + GTC + GGA + GGT + GGG + GGC + GCA + GCT + GCG + GCC + CAA + CAT + CAG + CAC + CTA + CTT + CTG + CTC + CGA + CGT + CGG + CGC + CCA + CCT + CCG + CCC + cnv_loss_avg + cnv_gain_avg + open_chromatin_avg, data=df_learn1, trace=0)
summary(forw1)


# Make predictions on the test set
y_pred2_test_sub <- predict(model2_sub, newdata = df_test2)
y_pred2_val_sub <- predict(model2_sub, newdata = df_val2)

# Calculate performance metrics
mse_test2_sub <- mse(y_test2, y_pred2_test_sub)
r2_test2_sub <- 1 - sum((y_test2 - y_pred2_test_sub)^2) / sum((y_test2 - mean(y_test2))^2)

mse_val2_sub <- mse(y_val2, y_pred2_val_sub)
r2_val2_sub <- 1 - sum((y_val2 - y_pred2_val_sub)^2) / sum((y_val2 - mean(y_val2))^2)


# Visualize predictions vs actual values
ggplot(data.frame(actual = y_test2_sub, predicted = y_pred2_sub), aes(x = actual, y = predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Predicted vs Actual Gene Expression",
       x = "Actual Gene Expression",
       y = "Predicted Gene Expression") +
  theme_minimal()
