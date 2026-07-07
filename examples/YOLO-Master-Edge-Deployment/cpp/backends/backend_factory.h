#pragma once

#include <memory>
#include <string>

#include "backend.h"

std::unique_ptr<Backend> create_backend(const std::string& backend);

