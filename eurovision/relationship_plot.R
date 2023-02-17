library(ggplot2)
library(tidyverse)
library(hrbrthemes)
library(GGally)
library(viridis)
library(ggrepel)
library(ggalt)

df <- read_csv("data/top_relationships.csv") %>%
        as.data.frame() %>% 
        select(from_country_name, country_name, votes_x, votes_y, group, num_years) %>%
        mutate(num = factor(c(15:11, 1:5, 6:10))) %>%
        mutate(group = factor(group, levels = c("haters", "unequals", "lovers"))) %>% 
        mutate(num = fct_reorder(num, group, votes_x))

df
#plot(df$votes_x, df$votes_y, pch = 19, cex = 2)



ggplot(df, aes(x = votes_x, xend = votes_y, y = num, color = group)) +
    geom_dumbbell() +
    #geom_point(size = 1) +
    theme_classic() +
    geom_text(color="black", size=2, hjust=-0.5,
                  aes(x=votes_x, label=country_name))+
    geom_text(aes(x=votes_y, label=from_country_name), 
                  color="black", size=2, hjust=1.5) +
    theme(
        axis.text.y = element_blank(),
        axis.line.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.title.y = element_blank()
    ) +
    scale_color_viridis(discrete=TRUE) +
    # x axis limits
    xlim(-3, 13) +
    xlab("Average (over the years) received vote")
    #coord_flip()

ggsave("plots/relationship_plot.jpg", width = 8, height = 4.5)


p <- ggparcoord(data = df, groupColumn = 5,
    columns = 3:4,
    scale = "globalminmax",
    showPoints = TRUE,
  ) +
scale_color_viridis(discrete=TRUE) +
theme_ipsum()+
theme(
    plot.title = element_text(size=10)
) +
geom_text(data = df %>%
              select(country_name, votes_x) %>%
              mutate(x = 1,
                     y = votes_x),
            aes(x = x, y = y, label = country_name),
            hjust = 1.1,
            inherit.aes = FALSE) +
geom_text(data = df %>%
            select(from_country_name, votes_y) %>%
            mutate(x = 2,
                     y = votes_y),
            aes(x = x, y = y, label = from_country_name),
            hjust = -0.1,
            inherit.aes = FALSE)
p
  # optional: remove "carName" from x-axis labels
  scale_x_discrete(labels = function(x) c("", x[-1]))

ggsave("plots/relationship_plot.jpg", p, width = 6, height = 4)




ggplot(aes(x = variable, y = value, label = country_name, group = group)) +
  geom_line() +
  geom_text(data = .,
            hjust = 1.1) +
  scale_x_discrete(labels = function(x) c("", x[-1]))

ggsave("plots/relationship_plot.jpg", p, width = 10, height = 5)


geom_text_repel(data = df
                  aes(x = variable, y = value, label = carName),
                  xlim = c(NA, 1))

ggsave("plots/relationship_plot.jpg", p, width = 10, height = 5)


