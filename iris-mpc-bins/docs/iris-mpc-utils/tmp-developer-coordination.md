Temporary file for communication of work done and help needed between two devs working in opposite time zones.

# SW
added test cases complex-tc1 through tc12 (tc7 was omitted).
test cases are documented in iris-mpc-bins/docs/ ... / service-client-test-cases.md
also added a docs/ folder to help claude not have to do repository exploration
changed service client to delete IrisSerialIds upon exit, to make the tests repeatable.
will test against localstack today.
