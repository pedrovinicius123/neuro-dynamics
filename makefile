CC ?= gcc
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic
SOURCES = neura.c mtrx_utils.c logging/logger.c

compile:
	$(CC) $(CFLAGS) $(SOURCES) -o neura -lpthread -lm

test: compile tests/test_neura
	./tests/test_neura
	python3 -m unittest discover -s tests -p 'test_*.py'

tests/test_neura: tests/test_neura.c neura.c snn_lif_stdp.h mtrx_utils.c threads/layer_thread.c threads/layer_thread.h logging/logger.c logging/logger.h
	$(CC) $(CFLAGS) -Werror -g -DNEURA_NO_MAIN tests/test_neura.c mtrx_utils.c threads/layer_thread.c logging/logger.c -o tests/test_neura -lpthread -lm

clean:
	rm -f neura tests/test_neura
