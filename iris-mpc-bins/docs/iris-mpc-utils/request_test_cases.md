other test cases

# enrollment
batch 1
enroll -> should show that is_match=false

batch 2
enroll -> should show is_match=true

batch 3
delete -> should show success = true

batch 4
delete -> should show failed

batch 5
enroll -> should show is_match=false



# reset_update
batch 1
enroll -> show is_match=false

batch 2
enroll -> show is_match=true

batch 3
reset_update the serial id from batch 1 with a different iris code

batch 4
enroll the iris code from batch 1 again -> show that is_match=false


# reauth
batch 1
enroll

batch 2
delete

batch 3
reauth -> should fail

batch 4
enroll again

batch 5 reauth -> succeed

# reset_check
same flow as reauth test case
