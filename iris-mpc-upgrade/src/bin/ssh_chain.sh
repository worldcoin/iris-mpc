CERT_PATH="./nginx/cert"

rm -rf $CERT_PATH/*.key
rm -rf $CERT_PATH:/*.pem
rm -rf $CERT_PATH/*.srl


# 1. Generate CA's private key and self-signed certificate
openssl req -x509 -newkey rsa:4096 -days 365 -nodes -keyout $CERT_PATH/ca-key.pem -out $CERT_PATH/ca-cert.pem -subj "/C=DE/ST=Berlin/L=Berlin/O=TFH/OU=Privacy/CN=*.2.stage.smpcv2.worldcoin.dev/emailAddress=carlo.mazzaferro@toolsforhumanity.com"

echo "CA's self-signed certificate"
openssl x509 -in $CERT_PATH/ca-cert.pem -noout -text

# 2. Generate web server's private key and certificate signing request (CSR)
openssl req -newkey rsa:4096 -nodes -keyout $CERT_PATH/server-key.pem -out $CERT_PATH/server-req.pem -subj "/C=DE/ST=Berlin/L=Berlin/O=Worldcoin/OU=Computer/CN=*.2.stage.smpcv2.worldcoin.dev/emailAddress=carlo.mazzaferro@toolsforhumanity.com"

# 3. Use CA's private key to sign web server's CSR and get back the signed certificate
openssl x509 -req -in $CERT_PATH/server-req.pem -days 60 -CA $CERT_PATH/ca-cert.pem -CAkey $CERT_PATH/ca-key.pem -CAcreateserial -out $CERT_PATH/server-cert.pem -extfile $CERT_PATH/server-ext.cnf

echo "Server's signed certificate"
openssl x509 -in $CERT_PATH/server-cert.pem -noout -text
