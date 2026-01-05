# GPU Iris MPC Deployment in current stage

The application right now has issues with DB loading. To run the app it is necessary to truncate tables in the dbs in all 3 parties.

To do so, please deploy the pod in `/deploy/db-cleaner-helper-pod.yaml` in every party and run the following command putting appropriate DB URL and party id in it beforehand:

```bash
apt update && apt install -y postgresql-client && psql -H <DB_URL> -c 'SET search_path TO "SMPC_stage_{0,1,2}"; TRUNCATE irises, results, sync;'
```

# Application Upgrade Documentation

This document provides a step-by-step guide on how to upgrade the application deployed using ArgoCD. The application configuration is located in the `deploy/stage/mpc1-stage`, `deploy/stage/mpc2-stage`, and `deploy/stage/mpc3-stage` directories. Each directory contains a `values-iris-mpc.yaml` file, which includes the deployment configuration for the respective clusters: `mpc1-stage`, `mpc2-stage`, and `mpc3-stage`, and common value file placed in `deploy/stage/values-common-gpu-iris-mpc.yaml`

## Prerequisites

-  Access to the Git repository where the deployment configuration is stored.
-  Permissions to modify and commit changes to the repository.
-  The URL of the new Docker image that you want to deploy.

## Upgrade Steps

### 1. Clone the Repository

First, clone the repository to your local machine if you haven't already:

```sh
git clone git@github.com:worldcoin/gpu-iris-mpc.git
cd gpu-iris-mpc
```
### 2. Update the Image in Configuration Files

For each cluster (`mpc1-stage`, `mpc2-stage`, `mpc3-stage`), you need to update the `values-gpu-iris-mpc.yaml` file with the new image URL.

#### Update deployment

1. Navigate to the `deploy/stage` directory:

    ```sh
    cd deploy/stage/
    ```

2. Open the `values-common-gpu-iris-mpc.yaml` file in a text editor and locate the `image:` parameter.

3. Update the `image:` parameter with the new image URL:

    ```yaml
    image: <new-image-url>
    ```

4. Save the file.

### 3. Commit and Create a Pull Request

After updating the image URL in all necessary files, commit the changes and create a Pull Request (PR):

1. Stage the changes:

    ```sh
    git add deploy/stage/values-common-gpu-iris-mpc.yaml
    ```

2. Commit the changes with a meaningful message:

    ```sh
    git commit -m "Update image URL for gpu-iris-mpc deployment"
    ```

3. Push the changes to a new branch:

    ```sh
    git push origin <new-branch-name>
    ```

4. Create a Pull Request (PR) from your branch in the repository's web interface. Ensure that you provide a clear description of the changes and the reason for the update.

5. Wait for the PR to be reviewed and approved by the designated reviewers.

6. Once the PR is approved, merge it into the main branch.

7. After merging the PR, ArgoCD will automatically detect the configuration updates within 5 minutes, pull the updated configuration, and start the upgrade process. 
