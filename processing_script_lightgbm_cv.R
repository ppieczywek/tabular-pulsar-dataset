library(tidyverse)
library("Cairo")
pacman::p_load('dplyr', 'tidyr', 'gapminder',
               'ggplot2',  'ggalt',
               'forcats', 'R.utils', 'png',
               'grid', 'ggpubr', 'scales',
               'bbplot')

library("ROSE")
library("lightgbm")

train.data <- read.csv(unz("playground-series-s3e10.zip", "train.csv"))


train.data %>%
  select(-id) %>%
  pivot_longer(!Class,
               names_to = "variable",
               values_to = "value") %>%
  group_by(variable,Class) %>%
  summarise(mean = mean(value),
            sd = sd(value),
            min = min(value),
            max = max(value),
            count = n()) -> train.data.summary

print(train.data.summary)

train.data %>%
  select(-id) %>%
  pivot_longer(!Class,
               names_to = "variable",
               values_to = "value") %>%
  group_by(variable) %>%
  mutate(w = cut_width(value, width = (max(value)-min(value))/ 30,
                       labels = seq(1,31) * ((max(value)-min(value))/ 30))) %>%
  ungroup() -> plot.data

train.data %>%
  select(-id) %>%
  cor() -> correlations
print(correlations)

axis.label.font.size = 14
axis.title.font.size = 16
group.labels.font.size = 12
legend.font.size = 9
dodge.value <- 0.00
aspect.ratio <- 1.0

CairoWin()
p <- ggplot(data = plot.data,
             mapping = aes(x = value,
                           fill = as.factor(Class))) +
  geom_hline(yintercept = 0,
             linewidth = 1,
             colour="#333333") +
  geom_histogram(aes(y=..count../sum(..count..)),
                 alpha=0.8,
                 position = 'identity',
                 bins = 50) +
  scale_fill_manual(values=c("#669999", "#FFD264")) +
  bbc_style() +
  theme(aspect.ratio=aspect.ratio) +
  theme(axis.title = ggplot2::element_text(family = "TT Arial",
                                           size = axis.title.font.size,
                                           face = "bold",
                                           color = "#222222")) +
  theme(axis.title.y = element_text(margin = margin(r = 0, l = 0))) +
  theme(axis.text.y = ggplot2::element_text(size= axis.label.font.size)) +
  theme(axis.text.x = ggplot2::element_text(size= axis.label.font.size,
                                            angle = 0,
                                            hjust = 0.5,
                                            vjust = 0.0,
                                            margin=margin(t=0, b=10))) +
  theme(plot.margin = margin(t=0.0, r=1.0, b=0.0, l=1.0, "cm")) +
  theme(panel.spacing = unit(1, "lines")) +
  theme(panel.border = element_blank()) +
  facet_wrap(variable ~ .,  ncol = 4, scales='free')
plot(p)


train.data <- select(train.data, - id)
new.set <- ROSE(Class ~ ., data=train.data, seed=123)$data
x <- select(new.set, -Class)
y <- select(new.set, Class)

lgb.train = lgb.Dataset(data=as.matrix(x,rownames=FALSE, colnames= FALSE),
                       label=as.matrix(y,rownames=FALSE, colnames= FALSE))

lgb.grid = list(objective = "binary",
                metric = 'binary_logloss',
                boosting ="gbdt",
                learning_rate = 0.0005,
                n_estimators = 70000,
                max_depth = -1,
                reg_alpha = 3.5500117634027073,
                reg_lambda = 9.866866754487022e-07,
                num_leaves = 7,
                colsample_bytree = 0.6,
                subsample = 0.9070311680268052,
                subsample_freq = 5,
                min_child_samples = 50,
                verbosity = -1,
                is_unbalance = FALSE)

lgb.normalizedgini = function(preds, dtrain) {
  actual = getinfo(dtrain, "label")
  score  = NormalizedGini(preds,actual)
  return(list(name = "gini", value = score, higher_better = TRUE))
}

lgb.model.cv = lgb.cv(params = lgb.grid, 
                      data = lgb.train, 
                      num_threads = 1 , 
                      nrounds = 7000,
                      early_stopping_rounds = 50,
                      eval_freq = 20, 
                      eval = lgb.normalizedgini, 
                      nfold = 3, 
                      stratified = FALSE)

best.iter = lgb.model.cv$best_iter
lgb.model = lgb.train(params = lgb.grid, 
                      data = lgb.train, 
                      num_threads = 1 ,
                      nrounds = best.iter,
                      eval_freq = 20,
                      eval = lgb.normalizedgini)

test.data <- read.csv(unz("playground-series-s3e10.zip", "test.csv"))
submission <- read.csv(unz("playground-series-s3e10.zip", "sample_submission.csv"))
test.data <- select(test.data, -id)
prediction <- predict(lgb.model, 
                      dat = as.matrix(test.data,
                                      rownames=FALSE,
                                      colnames= FALSE))
submission$Class <- prediction

write.csv(submission, 
          file="my_submission.csv",
          row.names = FALSE)