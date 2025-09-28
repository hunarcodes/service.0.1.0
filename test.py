import asyncio
import aiohttp
import time

URL = "http://localhost:3000/v1/embed"  # change if needed
text = """Monday: Woke up late, ugh, hit snooze too many times. Coffee was cold by the time I got to it. Work was a mess—meetings all morning, barely got through emails. Boss dumped a new project on me, due Friday, of course. Lunch was a sad desk salad. Tried to hit the gym after work, but just did 20 mins on the treadmill and called it a day. Watched some random cooking show on Netflix, tried to recreate their pasta dish but burned the sauce. Bed by 11, scrolling X for way too long.\n\nTuesday: Actually got up on time, miracle. Made a smoothie, felt briefly like a health guru. Work was calmer, knocked out some code for the new project. Had a call with Sarah, she’s stressed about her move next month. Lunch was tacos from the food truck—spicy but worth it. Afternoon was a slog, debugging code that *should* work but didn’t. Left work late, skipped gym, grabbed a burger on the way home. Watched half a movie, fell asleep on the couch.\n\nWednesday: Overslept again, spilled coffee on my shirt, great start. Work was chaos—server crashed, spent hours fixing it. Project’s coming along, but I’m behind. Lunch was just chips from the vending machine, didn’t have time for more. Got into an argument with Mike over some dumb process thing, whatever. After work, met up with Alex for drinks. He’s got a new dog, super cute but hyper. Stayed out too late, head’s foggy now.\n\nThursday: Felt like garbage, too many beers last night. Dragged myself to work, powered through with energy drinks. Made decent progress on the project, might actually finish by tomorrow. Lunch was pizza, team ordered in. Had a good chat with Lisa about some new Rust libraries, she’s always got cool tech tips. Tried to go for a run after work, but it started raining, so I just did some push-ups at home. Watched a true crime doc, got creeped out, double-checked my locks before bed.\n\nFriday: Crunch time. Finished the project by some miracle, submitted it just before the deadline. Boss seemed happy, so that’s something. Lunch was a burrito, ate it way too fast. Afternoon was slow, just cleaning up some code and answering emails. Left work early, treated myself to ice cream. Went to the gym properly this time, felt good to lift. Evening was chill, played some video games and ordered takeout. Saw some funny posts on X about coding life, sent a few to Lisa.\n\nSaturday: Slept in, glorious. Made pancakes, they were actually decent. Spent the morning cleaning my apartment, it was a disaster. Went for a long walk in the park, weather was perfect. Grabbed coffee with Emma, she’s planning a trip to Japan, jealous. Afternoon was lazy, binged a new sci-fi series. Tried to read a book but kept checking X instead. Made stir-fry for dinner, not bad. Stayed up late gaming, probably a bad idea.\n\nSunday: Woke up groggy, too much gaming last night. Did laundry, finally. Went grocery shopping, spent way too much on snacks. Meal prepped some chicken and rice for the week, feeling semi-organized. Watched football with Jake, his team lost, he was salty. Spent the evening planning next week’s work, already dreading Monday. Scrolled X, saw some cool Docker tips, saved them for later. Bed early for once, let’s see if I can stick to it.
"""
PAYLOAD = {"text": text}
CONCURRENCY = 10  # how many duplicate requests

async def send_request(session, idx):
    start = time.perf_counter()
    async with session.post(URL, json=PAYLOAD) as resp:
        data = await resp.json()
        elapsed = (time.perf_counter() - start) * 1000
        print(f"Request {idx} finished in {elapsed:.2f} ms, got embedding dim={len(data['embedding'])}")
        return elapsed

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i) for i in range(CONCURRENCY)]
        timings = await asyncio.gather(*tasks)
        print("\n--- Stats ---")
        print(f"Total requests: {len(timings)}")
        print(f"Min: {min(timings):.2f} ms")
        print(f"Max: {max(timings):.2f} ms")
        print(f"Avg: {sum(timings)/len(timings):.2f} ms")

if __name__ == "__main__":
    asyncio.run(main())
