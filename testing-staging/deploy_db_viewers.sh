#!/usr/bin/env bash


kubectx CORRECT_CONTEXT_HERE
kubectl apply -f db-viewer.yaml -n ampc-hnsw

kubectx CORRECT_CONTEXT_HERE
kubectl apply -f db-viewer.yaml -n ampc-hnsw

kubectx CORRECT_CONTEXT_HERE
kubectl apply -f db-viewer.yaml -n ampc-hnsw