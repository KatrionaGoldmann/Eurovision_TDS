library(ggplot2)
library(tidyverse)
library(hrbrthemes)
library(GGally)
library(viridis)
library(ggrepel)
library(ggalt)

df <- read_csv("data/top_relationships.csv") %>%
        select(from_country_name, country_name, votes_x, votes_y, group, num_years) %>%
        #mutate(num = factor(c(15:11, 1:5, 6:10))) %>%
        mutate(num = factor(1:nrow(.))) %>%
        mutate(group = factor(group, levels = c("haters", "unequals", "lovers"))) %>% 
        # group num factor by votes_y within group
        mutate(num = factor(num, levels = num[order(votes_y)])) 

df

ggplot(df, aes(x = votes_x, xend = votes_y, y = num, yend = num, color = group)) +
    geom_dumbbell(size_x = 1.6, size_xend = 1.6) +
    # change color of segment between points
    #geom_segment(aes(x = votes_x, xend = votes_y, y = num, yend = num), 
    #             color = "black", size = 0.2) +
    theme_classic() +
    geom_text(color="black", size=2.5, hjust=-0.5,
                  aes(x=votes_x, label=country_name))+
    geom_text(aes(x=votes_y, label=from_country_name), 
                  color="black", size=2.5, hjust=1.5) +
    theme(
        axis.text.y = element_blank(),
        axis.line.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.title.y = element_blank(),
        legend.title = element_blank(),
    ) +
    scale_color_viridis(discrete=TRUE) +
    xlim(-2, 12) +
    xlab("Average received vote") +
    guides(color = guide_legend(reverse = TRUE, override.aes = list(size=1.6))) +
    theme(legend.position = c(0.1, 0.9)) +
    # add space between plot and title
    theme(plot.title = element_text(margin = margin(t = 10, b = 15, unit = "pt"))) +
    ggtitle("The who likes (and doesn't like) who of Eurovision") +
    # add footnote
    annotate("text", x = 9, y = 2, label = "* The point next to a country shows the received \naverage vote from the country linked to it", size = 2.5) 

ggsave("plots/who_likes_who.jpg", width = 5.5, height = 4.5)
