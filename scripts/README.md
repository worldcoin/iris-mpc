## SMPCv2 Databases & Queues Purging

To make sure the scripts included in this repository are working properly, you need to have the following AWS profiles defined in your environment:

### SMPCv2
```yaml
[profile worldcoin-iam]
sso_start_url = https://worldcoin.awsapps.com/start
sso_region = us-east-1
sso_account_id = 033662022620
sso_role_name = AssumeSMPCV2Role

[profile worldcoin-smpcv2-vpc]
source_profile=worldcoin-iam
role_arn=arn:aws:iam::590183936500:role/smpcv2-cross-account-role

[profile worldcoin-smpcv2-1]
source_profile=worldcoin-iam
role_arn=arn:aws:iam::767397983205:role/smpcv2-cross-account-role

[profile worldcoin-smpcv2-2]
source_profile=worldcoin-iam
role_arn=arn:aws:iam::381492197851:role/smpcv2-cross-account-role

[profile worldcoin-smpcv2-3]
source_profile=worldcoin-iam
role_arn=arn:aws:iam::590184084615:role/smpcv2-cross-account-role
```

### Orb
```yaml
[profile worldcoin-stage]
sso_start_url = https://worldcoin.awsapps.com/start
sso_region = us-east-1
sso_account_id = 510867353226
sso_role_name = PowerUserAccess
```
