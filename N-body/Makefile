CPPFLAGS=

main: $(patsubst %.c,%.o,$(wildcard *.c)) $(patsubst %.cc,%.o,$(wildcard *.cc))
	g++ -std=c++11 $+ -o $@

%.o: %.cc Makefile
	g++ -c $(CPPFLAGS) $< -o $@

%.o: %.c Makefile
	g++ -c $(CPPFLAGS) $< -o $@
