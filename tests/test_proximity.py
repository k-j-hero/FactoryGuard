import unittest

from factoryguard.proximity import ProximityEngine, Track, normalized_proximity


def pair(x=52, person_id=1, forklift_id=2):
    return [Track(person_id, "person", (45, 30, 55, 50), 0.9),
            Track(forklift_id, "forklift", (x - 5, 30, x + 5, 50), 0.9)]


class ProximityTests(unittest.TestCase):
    def engine(self):
        return ProximityEngine(10, 0.08, 0.11, 0.3, 0.2)

    def test_metric_and_resolution_invariance(self):
        p, f = pair(60)
        value = normalized_proximity(p, f, 100, 100)
        scaled = [Track(t.track_id, t.label, tuple(x * 2 for x in t.bbox), t.confidence) for t in (p, f)]
        self.assertAlmostEqual(value, 10 / (20000 ** 0.5))
        self.assertAlmostEqual(value, normalized_proximity(*scaled, 200, 200))

    def test_short_transient_is_rejected(self):
        e = self.engine()
        e.update(0, pair(), 100, 100)
        _, active, closed = e.update(1, pair(90), 100, 100)
        self.assertFalse(active)
        self.assertEqual(closed + e.finish(), [])

    def test_confirmation_hysteresis_and_separation(self):
        e = self.engine()
        for frame in range(3):
            _, active, closed = e.update(frame, pair(), 100, 100)
        self.assertEqual(active, {(1, 2)})
        # 0.099 lies between enter/exit, so the same event remains active.
        self.assertEqual(e.update(3, pair(64), 100, 100)[1], {(1, 2)})
        event = e.update(4, pair(90), 100, 100)[2][0]
        self.assertEqual((event.start_frame, event.end_frame, event.confirmed_frame), (0, 3, 2))
        self.assertAlmostEqual(event.end_sec, 0.4)
        self.assertEqual(event.close_reason, "separated")

    def test_missing_frames_do_not_confirm(self):
        e = self.engine()
        e.update(0, pair(), 100, 100)
        e.update(1, [], 100, 100)
        e.update(2, [], 100, 100)
        self.assertFalse(e.update(3, pair(), 100, 100)[1])
        self.assertEqual(e.update(4, pair(), 100, 100)[1], {(1, 2)})
        event = e.finish()[0]
        self.assertAlmostEqual(event.observed_close_sec, 0.3)

    def test_track_loss_closes_at_last_observation(self):
        e = self.engine()
        for frame in range(3):
            e.update(frame, pair(), 100, 100)
        self.assertEqual(e.update(3, [], 100, 100)[2], [])
        self.assertEqual(e.update(4, [], 100, 100)[2], [])
        event = e.update(5, [], 100, 100)[2][0]
        self.assertEqual(event.end_frame, 2)
        self.assertEqual(event.close_reason, "track_lost")

    def test_separate_pairs_and_eof_flush(self):
        e = self.engine()
        for frame in range(3):
            e.update(frame, pair() + [Track(3, "person", (45, 30, 55, 50), 0.8)], 100, 100)
        events = e.finish()
        self.assertEqual(len(events), 2)
        self.assertEqual(len({event.event_id for event in events}), 2)
        self.assertEqual(e.finish(), [])

    def test_reentry_creates_new_event(self):
        e = self.engine()
        for frame in range(3):
            e.update(frame, pair(), 100, 100)
        first = e.update(3, pair(90), 100, 100)[2][0]
        for frame in range(4, 7):
            e.update(frame, pair(), 100, 100)
        second = e.finish()[0]
        self.assertNotEqual(first.event_id, second.event_id)
        self.assertEqual(second.start_frame, 4)

    def test_no_tracks_and_no_skipped_frames(self):
        e = self.engine()
        self.assertEqual(e.update(0, [], 100, 100), ({}, set(), []))
        with self.assertRaises(ValueError):
            e.update(2, [], 100, 100)


if __name__ == "__main__":
    unittest.main()
