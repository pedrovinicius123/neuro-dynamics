CC ?= gcc
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic
SOURCES = neura.c mtrx_utils.c threads/layer_thread.c logging/logger.c audio/astream.c

compile:
	$(CC) $(CFLAGS) $(SOURCES) -o neura -lpthread -lm

test: tests/test_neura
	./tests/test_neura
	python3 -m unittest discover -s tests -p 'test_*.py'

tests/test_neura: tests/test_neura.c neura.c mtrx_utils.c
	$(CC) $(CFLAGS) -Werror -g -DNEURA_NO_MAIN tests/test_neura.c mtrx_utils.c -o tests/test_neura -lm

clean:
	rm -f neura tests/test_neura
