
UNAME=$(shell uname -s)

ifeq ($(UNAME),Darwin)
	CC=clang
else
	CC=gcc
endif

CCFLAGS=-g -O3
LDFLAGS=-lgsl -lgslcblas -lm

PYCFLAGS=$(shell python3-config --cflags)
PYLDFLAGS=$(shell python3-config --ldflags)
CCFLAGS+=$(PYCFLAGS) -Isrc/
LDFLAGS+=$(PYLDFLAGS)

ifeq ($(UNAME),Darwin)
	LDFLAGS+=-undefined dynamic_lookup -bundle
endif

RM=rm -rf
MKDIR=mkdir -p

SRCS=$(wildcard src/*.c)
OBJS=$(subst src/,obj/,$(subst .c,.o,$(SRCS)))

OUT=estimate.so

.PHONY : all
all : $(OUT)

.PHONY: clean
clean:
	@$(RM) obj/

.PHONY: distclean
distclean:
	@$(RM) obj/
	@$(RM) $(OUT)


$(OUT): $(OBJS)
	$(CC) -o $(OUT) $^ $(LDFLAGS)

obj/%.o: src/%.c
	@$(MKDIR) obj/
	$(CC) $(CCFLAGS) -o $@ -c $<
