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

# SW
I made a branch called sw/service_client_6 where error handling is improved and the monolithic exec() function now delegates some processing to a state machine. Logs have useful information so that an AI assistant (or human) can compare with the logs for iris-mpc-hawk. Submitting requests all at once seems to break localstack by making the iris-mpc-hawk nodes disagree on the next item in the batch.

Interesting observations:
- I sent a duplicate deletion in i  think complex-tc1.toml and the servers never saw it. though this was before i switched to sending requests one at a time. this may not be worth looking at further.
- the complex-tc2.toml has mirrored iris codes and these all get non-matching uniqueness results, which means the mirroring detection did not work 

# MC (Feb 20)

I played with sw/service_client_6. Two things:
- Batch processing had further bugs w.r.t msg_counter; it didn't get incremented when skipping modifications for the same serial id; this made some batches stuck
- Claude noted there are separate SQS response queues for each request type so I had it change the client to check all these queues for replies.
- There was an issue with localstack SQS ordering about a year ago, but it should be fixed in the version we're using. Worth checking for a stale image. Details on Slack


# SW
addressing the review. i deployed main to the dev cluster so we can test the service client against it some. and regarding the tokio thread pool, the genesis testing is half done. i tested it with tokio workers restricted to one numa node and it was a little slower. i will try restricting it to cores on both nodes next. 
