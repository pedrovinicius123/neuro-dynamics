compile:
	gcc neura.c mtrx_utils.c threads/layer_thread.c logging/logger.c audio/astream.c -o neura -lpthread -lm
