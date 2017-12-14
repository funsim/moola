class EventCounter(dict):
    ''' A simple class for counting events. '''

    def increment(self, event_name):
        try: 
            self[event_name] += 1
        except KeyError:
            self[event_name] = 1

    def __str__(self):
        print("")
        print("="*20 + "Moola Events" + "="*20)
        print("")
        s = []

        row_format ="{:>25}" * 2
        for k, v in self.items():
            s.append(row_format.format(k, v))

        return "\n".join(s)


events = EventCounter()
