mpctl
===============

Bash application to work with a **local** IRIS-MPC CPU network.

What is mpctl ?
--------------------------------------

mpctl is a bash utility that allows a developer to work with a local IRIS-MPC network.

Why mpctl ?
--------------------------------------

Developers need to spin up small local throwaway networks.

Who uses mpctl ?
--------------------------------------

Command line friendly folks such as developers, testers, evaluators ... etc.

How to install mpctl ?
--------------------------------------

```
# Note: Ensure that `iris-mpc` monorepo has also been cloned into WORKING_DIRECTORY.

cd WORKING_DIRECTORY
git clone https://github.com/siajsal/mpctl.git
```

How to use mpctl ?
--------------------------------------

```
# To activate mpctl commands.
source WORKING_DIRECTORY/iris-mpc/tools/mpctl/activate

# To list available commands.
mpctl-ls

# To view help for a supported command.
mpctl-[CMD] help
```

What recipes can you do with mpctl ?
--------------------------------------

See [recipes](docs/recipe-local-net.md).

What can you do with mpctl ?
--------------------------------------

The mpctl utility can be used to:

- Control a network running in Docker.
- Control a network running on bare metal with Dockerised services
- Launch jobs for interacting with Dockerised postgres or localstack services.
- Launch jobs for testing.
