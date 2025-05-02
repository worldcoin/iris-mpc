#!/usr/bin/env bash


kubectx CORRECT_CONTEXT_HERE
kubectl apply -f db-init-pod.yaml -n ampc-hnsw

kubectx CORRECT_CONTEXT_HERE
kubectl apply -f db-init-pod-1.yaml -n ampc-hnsw

kubectx CORRECT_CONTEXT_HERE
kubectl apply -f db-init-pod-2.yaml -n ampc-hnsw