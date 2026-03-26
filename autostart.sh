sleep 5
cd /home/qianli/zkw_nuc06/qianli_vision
screen \
    -L \
    -Logfile logs/$(date "+%Y-%m-%d_%H-%M-%S").screenlog \
    -d \
    -m \
    bash -c "/home/qianli/zkw_nuc06/qianli_vision/build/hero /home/qianli/zkw_nuc06/qianli_vision/configs/hero.yaml"
