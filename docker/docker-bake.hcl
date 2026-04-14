# docker/docker-bake.hcl — declarative multi-image build with dependency graph
#
# Usage:
#   docker buildx bake -f docker/docker-bake.hcl                  # build cap-y:base
#   docker buildx bake -f docker/docker-bake.hcl cap-y-nvidia     # builds base→open→nvidia
#   docker buildx bake -f docker/docker-bake.hcl all              # base + default + open + nvidia
#   docker buildx bake -f docker/docker-bake.hcl full             # all + dev
#   CUDA_ARCH_BIN=9.0 docker buildx bake -f docker/docker-bake.hcl all
#
# Bake resolves the build graph via `contexts`: when cap-y-nvidia declares
#   contexts = { "cap-y:open" = "target:cap-y-open" }
# Bake knows to build cap-y-open first and pipe its output directly as the
# FROM source — no intermediate registry push needed.

variable "CUDA_ARCH_BIN" {
  default = ""   # auto-detected by the build ARG default inside Dockerfile.base
}

variable "WITH_DEMOGRASP" {
  default = "1"
}

# BAKE_LOCAL_PLATFORM is a Docker Bake built-in: the host platform (e.g. linux/amd64).
# Declaring it here lets users override for cross-builds:
#   TARGETPLATFORM=linux/arm64 docker buildx bake -f docker/docker-bake.hcl all
variable "REGISTRY" {
  default = ""   # e.g. "sevapru/cap-y" — set via --push in build.sh
}

variable "BAKE_LOCAL_PLATFORM" {
  default = "linux/amd64"
}

variable "TARGETPLATFORM" {
  default = BAKE_LOCAL_PLATFORM
}

# Derive TARGETARCH from the platform string (linux/amd64 → amd64, linux/arm64 → arm64).
# Passed explicitly as a build arg because the rootless docker driver does not
# auto-inject TARGETARCH even when platforms is set.
variable "TARGETARCH" {
  default = element(split("/", TARGETPLATFORM), 1)
}

function "remote_tags" {
  params = [name]
  result = REGISTRY != "" ? ["${REGISTRY}:${name}-${TARGETARCH}"] : []
}

function "remote_tags_latest" {
  params = [name]
  result = REGISTRY != "" ? ["${REGISTRY}:${name}-${TARGETARCH}", "${REGISTRY}:latest-${TARGETARCH}"] : []
}


# ── Groups ────────────────────────────────────────────────────────────────────

group "default" {
  targets = ["cap-y-base"]
}

# base + cuRobo + open-source robotics stack + Isaac ROS
group "all" {
  targets = ["cap-y-base", "cap-y-default", "cap-y-open", "cap-y-nvidia"]
}

# all + dev (large image, caches retained)
group "full" {
  targets = ["cap-y-base", "cap-y-default", "cap-y-open", "cap-y-nvidia", "cap-y-dev"]
}

# ── Targets ───────────────────────────────────────────────────────────────────

target "cap-y-base" {
  context    = "."
  dockerfile = "docker/Dockerfile.base"
  tags       = concat(["cap-y:base"], remote_tags("base"))
  platforms  = [TARGETPLATFORM]
  args = {
    TARGETARCH     = TARGETARCH
    CUDA_ARCH_BIN  = CUDA_ARCH_BIN
    WITH_DEMOGRASP = WITH_DEMOGRASP
  }
}

target "cap-y-default" {
  context    = "."
  dockerfile = "docker/Dockerfile"
  tags       = concat(["cap-y:default"], remote_tags("default"))
  platforms  = [TARGETPLATFORM]
  args = {
    TARGETARCH = TARGETARCH
  }
  contexts = {
    "cap-y:base" = "target:cap-y-base"
  }
}

target "cap-y-open" {
  context    = "."
  dockerfile = "docker/Dockerfile.open"
  tags       = concat(["cap-y:open"], remote_tags_latest("open"))
  platforms  = [TARGETPLATFORM]
  args = {
    TARGETARCH    = TARGETARCH
    CUDA_ARCH_BIN = CUDA_ARCH_BIN
  }
  contexts = {
    "cap-y:base" = "target:cap-y-base"
  }
}

target "cap-y-nvidia" {
  context    = "."
  dockerfile = "docker/Dockerfile.nvidia"
  tags       = concat(["cap-y:nvidia"], remote_tags("nvidia"))
  platforms  = [TARGETPLATFORM]
  args = {
    TARGETARCH = TARGETARCH
  }
  contexts = {
    "cap-y:open" = "target:cap-y-open"
  }
}

target "cap-y-dev" {
  context    = "."
  dockerfile = "docker/Dockerfile.dev"
  tags       = concat(["cap-y:dev"], remote_tags("dev"))
  platforms  = [TARGETPLATFORM]
  args = {
    TARGETARCH    = TARGETARCH
    CUDA_ARCH_BIN = CUDA_ARCH_BIN
  }
  contexts = {
    "cap-y:base" = "target:cap-y-base"
  }
}
