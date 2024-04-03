# Parsley

## Usage
```shell
$ python -m app.model -h
usage: model.py [-h] [--train] [--predict IMAGE_PATH] [--epochs EPOCHS] [--train-path TRAIN_PATH]

Train a model on specified images or predict using an existing model.

options:
  -h, --help            show this help message and exit
  --train               Download images and train the model.
  --predict IMAGE_PATH  Predict an image. Requires path to image.
  --epochs EPOCHS       Number of epochs to train the model.
  --train-path TRAIN_PATH
                        Path to store training images.
```

## Using knative with camel-k

```shell
# Create knative service
$ kn service create parsley --image quay.io/bzlotnik/parsley:latest --port 8080 --namespace my-namespace --env-from secret:aws-creds --env AWS_DEFAULT_REGION=<region> RESULT_BUCKET=<bucket>
```

```shell
$ cat aws-creds.properties
sqs.secretKey=
sqs.accessKey=
sqs.queueName=
sqs.region=
```

$ oc create secret generic aws-creds --from-file=aws-creds.properties
```

```shell
# Create KameletBinding
$ cat <<EOF | oc apply -f -
apiVersion: camel.apache.org/v1alpha1
kind: KameletBinding
metadata:
  annotations:
    trait.camel.apache.org/mount.configs: 'secret:aws-creds'
  name: aws-sqs-source-binding
  namespace: my-namespace
spec:
  sink:
    ref:
      apiVersion: serving.knative.dev/v1
      kind: Service
      name: parsley
  source:
    properties:
      accessKey: '{{sqs.accessKey}}'
      queueNameOrArn: '{{sqs.queueName}}'
      region: '{{sqs.region}}'
      secretKey: '{{sqs.secretKey}}'
    ref:
      apiVersion: camel.apache.org/v1alpha1
      kind: Kamelet
      name: aws-sqs-source
EOF
```

