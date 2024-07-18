# Application Upgrade Documentation

This document provides a step-by-step guide on how to upgrade the application deployed using ArgoCD. The application configuration is located in the `deploy/stage/mpc1-stage`, `deploy/stage/mpc2-stage`, and `deploy/stage/mpc3-stage` directories. Each directory contains a `values-gpu-iris-mpc.yaml` file, which includes the deployment configuration for the respective clusters: `mpc1-stage`, `mpc2-stage`, and `mpc3-stage`.

## Prerequisites

-  Access to the Git repository where the deployment configuration is stored.
-  Permissions to modify and commit changes to the repository.
-  The URL of the new Docker image that you want to deploy.

## Upgrade Steps

### 1. Clone the Repository

First, clone the repository to your local machine if you haven't already:

```sh
git clone <repository-url>
cd <repository-directory>
```
### 2. Update the Image in Configuration Files

For each cluster (`mpc1-stage`, `mpc2-stage`, `mpc3-stage`), you need to update the `values-gpu-iris-mpc.yaml` file with the new image URL.

#### Update `mpc1-stage`

1. Navigate to the `mpc1-stage` directory:

    ```sh
    cd deploy/stage/mpc1-stage
    ```

2. Open the `values-gpu-iris-mpc.yaml` file in a text editor and locate the `image:` parameter.

3. Update the `image:` parameter with the new image URL:

    ```yaml
    image: <new-image-url>
    ```

4. Save the file.

#### Update `mpc2-stage`

1. Navigate to the `mpc2-stage` directory:

    ```sh
    cd deploy/stage/mpc2-stage
    ```

2. Open the `values-gpu-iris-mpc.yaml` file and locate the `image:` parameter.

3. Update the `image:` parameter with the new image URL:

    ```yaml
    image: <new-image-url>
    ```

4. Save the file.

#### Update `mpc3-stage`

1. Navigate to the `mpc3-stage` directory:

    ```sh
    cd deploy/stage/mpc3-stage
    ```

2. Open the `values-gpu-iris-mpc.yaml` file and locate the `image:` parameter.

3. Update the `image:` parameter with the new image URL:

    ```yaml
    image: <new-image-url>
    ```

4. Save the file.

### 3. Commit and Create a Pull Request

After updating the image URL in all necessary files, commit the changes and create a Pull Request (PR):

1. Stage the changes:

    ```sh
    git add deploy/stage/mpc1-stage/values-gpu-iris-mpc.yaml
    git add deploy/stage/mpc2-stage/values-gpu-iris-mpc.yaml
    git add deploy/stage/mpc3-stage/values-gpu-iris-mpc.yaml
    ```

2. Commit the changes with a meaningful message:

    ```sh
    git commit -m "Update image URL for mpc1-stage, mpc2-stage, and mpc3-stage"
    ```

3. Push the changes to a new branch:

    ```sh
    git push origin <new-branch-name>
    ```

4. Create a Pull Request (PR) from your branch in the repository's web interface. Ensure that you provide a clear description of the changes and the reason for the update.

5. Wait for the PR to be reviewed and approved by the designated reviewers.

6. Once the PR is approved, merge it into the main branch.

7. After merging the PR, ArgoCD will automatically detect the changes, pull the updated configuration, and deploy the new image.
