from hz_linesearch import HZLineSearch

hz = HZLineSearch()

hz.set_print_level(3)
hz.print_parameters()

print "Running search"
hz.search()
