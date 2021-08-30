tb:
	tensorboard --logdir /logs/ --host 0.0.0.0 --port 8081

sync:
	git add -A
	git commit -m "auto-sync"
	git push