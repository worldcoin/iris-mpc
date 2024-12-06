## SMPCv2 Databases & Queues Purging

To make sure the scripts included in this repository are working properly, you need to have the following AWS profiles defined in your environment:

### SMPCv2
```yaml
[profile worldcoin-iam]
sso_start_url = https://worldcoin.awsapps.com/start
sso_region = us-east-1
sso_account_id = 033662022620
sso_role_name = AssumeSMPCV2Role

[profile worldcoin-smpcv-io-vpc]
source_profile=wc-iam
role_arn=arn:aws:iam::302263054573:role/smpcv2-cross-account-role

[profile worldcoin-smpcv-io-0]
source_profile=wc-iam
role_arn=arn:aws:iam::024848486749:role/smpcv2-cross-account-role

[profile worldcoin-smpcv-io-1]
source_profile=wc-iam
role_arn=arn:aws:iam::024848486818:role/smpcv2-cross-account-role

[profile worldcoin-smpcv-io-2]
source_profile=wc-iam
role_arn=arn:aws:iam::024848486770:role/smpcv2-cross-account-role
```

### Orb
```yaml
[profile worldcoin-stage]
sso_start_url = https://worldcoin.awsapps.com/start
sso_region = us-east-1
sso_account_id = 510867353226
sso_role_name = PowerUserAccess
```
