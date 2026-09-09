param([string]$Ip, [int]$Port = 22)
ssh -t -p $Port "ubuntu@$Ip" 'cd /infinite/common/yqy; bash init_YOLO.sh; exec bash'