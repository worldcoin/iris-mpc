aws \
 --profile $PROFILE \
 --region $REGION ec2 run-instances \
 --instance-type p5.48xlarge \
 --count 2 \
 --key-name ps-test-p5 \
 --image-id ami-0393966b92beee2d9 \
 --block-device-mapping 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,DeleteOnTermination=true,VolumeType=gp2}' \
 --network-interfaces "NetworkCardIndex=0,DeviceIndex=0,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=1,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=2,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=3,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=4,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=5,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=6,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=7,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=8,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=9,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=10,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=11,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=12,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=13,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=14,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=15,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=16,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=17,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=18,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=19,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=20,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=21,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=22,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=23,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=24,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=25,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=26,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=27,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=28,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=29,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=30,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
                      "NetworkCardIndex=31,DeviceIndex=1,Groups=sg-0895525324d0e6621,SubnetId=subnet-079d5d5d6356dec10,InterfaceType=efa" \
 --capacity-reservation-specification CapacityReservationTarget={CapacityReservationId=cr-077d7b182a8ec1b28}  \
 --instance-market-options '{"MarketType":"capacity-block"}' \
 --placement "GroupName=gpu-mpc, Tenancy=default" 