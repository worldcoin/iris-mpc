ENIS=$(aws ec2 describe-instances --instance-ids $1 --query 'Reservations[*].Instances[*].NetworkInterfaces[*].NetworkInterfaceId' --output text --profile $PROFILE --region $REGION)

for eni in $ENIS; do
    echo "Disabling source/dest check on $eni"
    aws ec2 modify-network-interface-attribute --network-interface-id $eni --no-source-dest-check --profile $REGION --region $REGION
done