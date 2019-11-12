import datetime
import unittest.mock

from absl.testing import absltest

import sheet_manager


class TestCycleDeadline(absltest.TestCase):

  def testAlwaysThursday(self):
    today = datetime.date.today()
    for _ in range(7):
      with unittest.mock.patch('datetime.date') as date:
        date.today.return_value = today
        deadline = sheet_manager.CycleDeadline()
        self.assertEqual(sheet_manager.THURSDAY, deadline.weekday())
        self.assertBetween((deadline - today).days, 10, 16)
      today += datetime.timedelta(days=1)


if __name__ == '__main__':
  absltest.main()
