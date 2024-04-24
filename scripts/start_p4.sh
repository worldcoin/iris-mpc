aws \
 --profile $PROFILE \
 --region $REGION ec2 run-instances \
 --instance-type p4d.24xlarge \
 --count 2 \
 --key-name ps-west2 \
 --image-id ami-0bcc00a005e32515b \
 --block-device-mapping 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,DeleteOnTermination=true,VolumeType=gp2}' \
 --network-interfaces "NetworkCardIndex=0,DeviceIndex=0,Groups=sg-034dbded92e0a123b,SubnetId=subnet-0f4ebd3a264d0dc0b,InterfaceType=efa" \
                      "NetworkCardIndex=1,DeviceIndex=1,Groups=sg-034dbded92e0a123b,SubnetId=subnet-0f4ebd3a264d0dc0b,InterfaceType=efa" \
                      "NetworkCardIndex=2,DeviceIndex=1,Groups=sg-034dbded92e0a123b,SubnetId=subnet-0f4ebd3a264d0dc0b,InterfaceType=efa" \
                      "NetworkCardIndex=3,DeviceIndex=1,Groups=sg-034dbded92e0a123b,SubnetId=subnet-0f4ebd3a264d0dc0b,InterfaceType=efa" \
 --capacity-reservation-specification CapacityReservationTarget={CapacityReservationId=cr-07684977c36fb3dc9}  \
 --instance-market-options '{"MarketType":"capacity-block"}' \
 --placement "GroupName=gpu-mpc, Tenancy=default" 