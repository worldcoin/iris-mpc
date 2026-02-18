Temporary file for communication of work done and help needed between two devs working in opposite time zones.

# SW
added test cases complex-tc1 through tc12 (tc7 was omitted).
test cases are documented in iris-mpc-bins/docs/ ... / service-client-test-cases.md
also added a docs/ folder to help claude not have to do repository exploration
changed service client to delete IrisSerialIds upon exit, to make the tests repeatable.
I tested a few of the test cases against localstack and found the uniqueness results were "not correlated". need to investigate further.

note that the iris.ndjosn and toml files can be viewed by running `scripts/run-service-client.sh -ls`. `-i` is the option for the .ndjson file. and to build the docker container, do `docker build -t hawk-server-local-build -f Dockerfile.dev.hawk .` and then to run localstack, `docker compose -f docker-compose.test.yaml up`

# SW
I made a branch called sw/sc_refactor where i made a version of the service client that i'm reasonably happy with. Rather than have N levels of abstraction which doesn't do much except make it hard access the code, i made a big loop with a few hashmaps. oh and i made an iterator of RequestBatchOptions. using this to test, my test cases passed, and nothing got stuck hanging around.

note that as we discussed, iris-mpc-hawk does not send responses for deletion requests unless there is a non-deletion request in the batch. and this may be a bug.

