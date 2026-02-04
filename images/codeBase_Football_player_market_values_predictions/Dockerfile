FROM busybox
WORKDIR /tmp/

COPY ../data/scraped-football-player-market-values/activation/activation_data.csv /tmp/codeBase/activation_data.csv
COPY ../code/Football_player_market_values_predictions/models/ /tmp/codeBase/
COPY ../code/Football_player_market_values_predictions/ANN_apply.py /tmp/codeBase/
COPY ../code/Football_player_market_values_predictions/OLS_apply.py /tmp/codeBase/
