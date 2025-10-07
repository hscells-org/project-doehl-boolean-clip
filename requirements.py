import pkg_resources

# Collect all installed distribution names
package_names = {dist.project_name for dist in pkg_resources.working_set}

# Sort and write to requirements.txt without version specifiers
sorted_packages = sorted(package_names)
filtered = filter(lambda x: "torch" not in x, sorted_packages)

with open('requirements.txt', 'w') as req_file:
    req_file.write("\n".join(filtered))