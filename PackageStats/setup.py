# -*- coding: utf-8 -*-

from distutils.core import setup

setup(
      name="MonPackageStats",
      packages=["PackageStats"],
      requires=["pandas"],
      package_dir={
          "PackageStats": "src/PackageStatsDir"
          }
      )
