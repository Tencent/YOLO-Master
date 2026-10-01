// Registry of loaded models (id -> WorkerPool). Loading is lazy-safe and idempotent.
#pragma once
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>
#include "worker_pool.hpp"

namespace yolomaster::server {

class ModelRegistry {
public:
    explicit ModelRegistry(const ServerConfig& cfg) : cfg_(cfg) {
        for (const auto& m : cfg.models) specs_[m.id] = m;
    }
    // Start the pool for `id` (spec from config, or an ad-hoc spec). Returns false + err when unknown.
    bool load(const std::string& id, std::string& err) {
        std::unique_lock<std::shared_mutex> lk(m_);
        if (pools_.count(id)) return true;
        auto it = specs_.find(id);
        if (it == specs_.end()) { err = "unknown model id: " + id; return false; }
        pools_[id] = std::make_unique<WorkerPool>(it->second, cfg_);
        return true;
    }
    bool add_spec(const ModelSpec& spec) {
        std::unique_lock<std::shared_mutex> lk(m_);
        if (specs_.count(spec.id)) return false;
        specs_[spec.id] = spec; return true;
    }
    bool unload(const std::string& id) {
        std::unique_ptr<WorkerPool> p;
        {
            std::unique_lock<std::shared_mutex> lk(m_);
            auto it = pools_.find(id);
            if (it == pools_.end()) return false;
            p = std::move(it->second); pools_.erase(it);
        }
        p->shutdown(2000);   // outside the lock: workers may be mid-inference
        return true;
    }
    // Shared pointer-free access: pools live until unload(); callers hold the shared lock for the
    // submit() call only, and results are delivered by the worker thread later.
    WorkerPool* get(const std::string& id) {
        std::shared_lock<std::shared_mutex> lk(m_);
        auto it = pools_.find(id);
        return it == pools_.end() ? nullptr : it->second.get();
    }
    std::vector<WorkerPool*> pools() {
        std::shared_lock<std::shared_mutex> lk(m_);
        std::vector<WorkerPool*> v; for (auto& kv : pools_) v.push_back(kv.second.get()); return v;
    }
    std::vector<ModelSpec> specs() const {
        std::shared_lock<std::shared_mutex> lk(m_);
        std::vector<ModelSpec> v; for (auto& kv : specs_) v.push_back(kv.second); return v;
    }
    bool has_spec(const std::string& id) const { std::shared_lock<std::shared_mutex> lk(m_); return specs_.count(id) > 0; }
    bool loaded(const std::string& id) const { std::shared_lock<std::shared_mutex> lk(m_); return pools_.count(id) > 0; }
    // ready = every preload model has a warmed worker (and none failed)
    bool ready(std::string& why) const {
        std::shared_lock<std::shared_mutex> lk(m_);
        for (auto& kv : specs_) {
            if (!kv.second.preload) continue;
            auto it = pools_.find(kv.first);
            if (it == pools_.end()) { why = kv.first + " not loaded"; return false; }
            if (it->second->failed()) { why = kv.first + ": " + it->second->init_error(); return false; }
            if (!it->second->ready()) { why = kv.first + " warming up"; return false; }
        }
        return true;
    }
    void shutdown_all(int drain_ms) {
        std::vector<std::unique_ptr<WorkerPool>> all;
        { std::unique_lock<std::shared_mutex> lk(m_); for (auto& kv : pools_) all.push_back(std::move(kv.second)); pools_.clear(); }
        for (auto& p : all) p->shutdown(drain_ms);
    }
private:
    ServerConfig cfg_;
    mutable std::shared_mutex m_;
    std::map<std::string, ModelSpec> specs_;
    std::map<std::string, std::unique_ptr<WorkerPool>> pools_;
};

} // namespace yolomaster::server
