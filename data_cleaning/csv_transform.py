#!/usr/bin/env python

'''
A class for csv transformation functions. Requires Python 2.x.

To run the test, use `python csv_transform.py test.csv` with test.csv in the
same directory; or, if you have allowed executable persmissions, you can use
`.\csv_transform.py test.csv`.
'''

import csv
import re
import datetime
import os
import sys
from copy import copy

class CSVTransform:

    def __init__(self, filepath):
        self.file = filepath
        self.row = 0
        self.columns = None

    def read(self, delimiter=",", quotechar='"', header=True):
        '''
        Returns an iterator for the csv file. Lazy reader for large data. If
        desired, delimiter and quotechar can be edited; default is comma
        delimited and double quotations (").
        '''
        with open(self.file, 'r') as f:
            reader = csv.DictReader(f, delimiter=delimiter, quotechar=quotechar)
            for row in reader:
                if self.row == 0:
                    self.columns = list(row.keys())
                    self.row += 1
                yield row

    def normalize_string(self, row, column, regex=r'\s+'):
        '''
        Breaks row[column] into regex matches, thereby removing anything not
        matching the regex, then joins to make a single space delimited string.
        Default behavior is set to match on non-whitespace; meaning all
        whitespace (including breaks, etc.) are removed, but punctuation
        remains. Will accept any regex instead of default '\s+' and will use
        for splitting instead.
        '''
        index = self.columns.index(column)
        row = copy(row)
        row[index] = ' '.join(re.split(regex, row[index]))
        return row

    def reference_swap(self, row, column, ref_file):
        '''
        Swaps row[column] for dictionary[row[column]]. Takes a 1:1 dictionary
        mapping instance value to desired swap value. Will only swap one-way,
        from the first column to the second.
        '''
        reference = self._create_reference(ref_file)
        index = self.columns.index(column)
        row = copy(row)
        if row[index] in reference:
            row[index] = reference[row[index]]
        return row

    def clean_dates(self, row, column, new_column, ref_file):
        '''
        Takes a row[column] date which may be variously formatted and pulls,
        if possible, the month, date, and year, returning it in ISO 8601 in the
        same column. Assumes US style date order for MM-DD (e.g. May 05
        not 05 May). Returns the row with edits to row[column] and a new column
        appended to the end called new_column.

        If no date is found, or an incomplete date is found, the value in
        row[column] is changed to '' and the old value is placed into
        row[new_column].
        '''
        months = self._create_reference(ref_file)
        if new_column not in self.columns:
            self.columns.append(new_column)
        row = copy(row)
        index = self.columns.index(column)
        digits = re.findall(r'[0-9]+', row[index])
        words = re.findall(r'[a-zA-Z]+', row[index])
        year, month, day = self._find_year_month_day(digits, words, months)
        if year and month and day:
            row[index] = '{y}-{m:02d}-{d:02d}'.format(y=year, m=month, d=day)
            row.append('')
        else:
            row.append(row[index])
            row[index] = ''
        return row

    def _find_year_month_day(self, digits, words, months):
        '''
        Filters found digits/words into date buckets, if appropriate. Returns
        a tuple of (year, month, day) as ints or None, depending on whether they
        are present.
        '''
        year, month, day = (None, None, None)
        for word in words:
            if word.lower() in months.keys():
                month = int(months[word.lower()])
        for number in digits:
            if len(number) == 4:
                year = int(number)
            elif len(number) <= 2 and month is None and int(number) < 12:
                # Assumes month will be first, as in US format
                month = int(number)
            elif len(number) <= 2 and day is None and int(number) < 31:
                # Assumes date will be second, as in US format
                day = int(number)
            elif len(number) == 2 and year is None:
                # If year is two digits and last, as in MM/DD/YY
                year = int('20' + str(number))
                if year > datetime.datetime.now().year:
                    year = int('19' + str(number))
        return (year, month, day)

    def _create_reference(self, ref_file):
        '''
        Creates a dictionary from a 1:1 reference csv, ref_file, which must
        be located in the same directory.
        '''
        reference = {}
        with open(ref_file, 'r') as f:
            reader = csv.reader(f)
            reader.next()
            for k, v in reader:
                reference[k] = v
        return reference

    def remove_word(self, row, column, word, replacement=''):
        '''
        Removes a word from a column.
        '''
        row[column] = row[column].replace(word, replacement)
        return row

    def write(self, row, filename, method='a'):
        '''
        Writes to file, default appends. If filename does not exist, creates
        file with self.columns as headers then writes row. If called with
        method = 'w', will overwrite existing file with row.
        '''
        if not os.path.exists(filename) and method is 'a':
            self.write(row, filename, 'w')
        with open(filename, method) as f:
            writer = csv.DictWriter(f, fieldnames = self.columns)
            if method == 'w':
                writer.writeheader()
            writer.writerow(row)

if __name__ == '__main__':
    filename = sys.argv[1]
    transform = CSVTransform(filename)
    for row in transform.read():
        row = transform.normalize_string(row, 'bio')
        row = transform.reference_swap(row, 'state', 'state_abbreviations.csv')
        row = transform.clean_dates(row, 'start_date', 'start_date_description',
                                    'months.csv')
        transform.write(row, 'solution.csv')
